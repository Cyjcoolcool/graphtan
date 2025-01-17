import networkx as nx
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
import pandas as pd
import numpy as np
import os
from os.path import join, exists
import pickle
from typing import Tuple
import pickle

class TemporalGraph():
    def __init__(self, data: pd.DataFrame, time_granularity: str, dataset_name: str):
        '''
        :param data: DataFrame- source, target, time, weight columns
        :param time_granularity: 'days', 'weeks', 'months', 'years' or 'hours'
        '''
        data['day'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).day)
        data['week'] = data['time'].apply(
            lambda timestamp: (datetime.utcfromtimestamp(timestamp)).isocalendar()[1])
        data['month'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).month)
        data['year'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).year)
        data['hour'] = data['time'].apply(lambda timestamp: (datetime.utcfromtimestamp(timestamp)).hour)
        if 'weight' not in data.columns:
            data['weight'] = 1

        if dataset_name == 'facebook':
            data = data[((data['year'] == 2006) & (data['month'] >= 8)) | (data['year'] > 2006)]

        elif dataset_name == 'enron_m':
            data = data[
                ((data['year'] < 2002) & (data['year'] >= 1999)) | ((data['year'] == 2002) & (data['month'] < 7))]

        self.data = data
        self.time_granularity = time_granularity
        self.time_columns, self.step = self._get_time_columns(time_granularity)
        self.static_graph = self.get_static_graph()
        # self.data['time_index'] = self.data.apply(self._get_time, axis=1)
        self.data.loc[:, 'time_index'] = self.data.apply(self._get_time, axis=1)


    def get_static_graph(self):

        g = nx.from_pandas_edgelist(self.data, source='source', target='target', edge_attr=['weight'],
                                    create_using=nx.MultiDiGraph())
        self.nodes = g.nodes()
        return g

    def filter_nodes(self, thresh: int = 5):
        nodes2filter = [node for node, degree in self.static_graph.degree() if degree < thresh]
        return nodes2filter

    def get_temporal_graphs(self, min_degree: int, mode: str = 'dynamic') -> dict:
        '''

        :param min_degree: int.  filter nodes with degree<min_degree in all time steps
        :param mode: if not 'dynamic', add all nodes to the current time step without edges
        :return: dictionary. key- time step, value- nx.Graph
        '''
        G = {}
        for t, time_group in self.data.groupby(self.time_columns):
            time_group = time_group.groupby(['source', 'target'])['weight'].sum().reset_index()
            g = nx.from_pandas_edgelist(time_group, source='source', target='target', edge_attr=['weight'],
                                        create_using=nx.DiGraph())
            if mode != 'dynamic':
                g.add_nodes_from(self.nodes)
            g.remove_nodes_from(self.filter_nodes(min_degree))
            G[self.get_date(t)] = g
        self.graphs = G
        return G

    def _get_time(self, x):
        if 'week' in self.time_columns:
            return datetime.strptime(f"{x.year}-W{x.week}" + '-1', "%Y-W%W-%w")
        elif 'hour' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=x.day, hour=x.hour)
        elif 'day' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=x.day)
        elif 'month' in self.time_columns:
            return datetime(year=x.year, month=x.month, day=1)
        elif 'year' in self.time_columns:
            return datetime(year=x.year, month=1, day=1)

    def get_date(self, t) -> datetime:
        time_dict = dict(zip(self.time_columns, t if type(t) == tuple else [t]))
        if self.time_granularity == 'hours':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=time_dict['day'],
                            hour=time_dict['hour'])
        elif self.time_granularity == 'days':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=time_dict['day'])
        elif self.time_granularity == 'months':
            return datetime(year=time_dict['year'], month=time_dict['month'], day=1)
        elif self.time_granularity == 'weeks':
            date_year = datetime(year=time_dict['year'], month=1, day=1)
            return date_year + timedelta(days=float((time_dict['week'] - 1) * 7))
        elif self.time_granularity == 'years':
            return datetime(year=time_dict['year'], month=1, day=1)
        else:
            raise Exception("not valid time granularity")

    @staticmethod
    def _get_time_columns(time_granularity: str):
        if time_granularity == 'hours':
            group_time = ['year', 'month', 'day', 'hour']
            step = timedelta(hours=1)
        elif time_granularity == 'days':
            group_time = ['year', 'month', 'day']
            step = timedelta(days=1)
        elif time_granularity == 'weeks':
            group_time = ['year', 'week']
            step = timedelta(weeks=1)
        elif time_granularity == 'months':
            group_time = ['year', 'month']
            step = relativedelta(months=1)
        elif time_granularity == 'years':
            group_time = ['year']
            step = relativedelta(years=1)
        else:
            raise Exception("not valid time granularity")
        return group_time, step


# def load_dataset(graph_df: pd.DataFrame, dataset_name: str, time_granularity: str) -> tuple[nx.Graph, TemporalGraph]:
def load_dataset(graph_df: pd.DataFrame, dataset_name: str, time_granularity: str) -> Tuple[nx.Graph, TemporalGraph]:
    # 函数实现

    '''

    :param graph_df:  DataFrame- source, target, time, weight columns
    :param dataset_name: name of the dataset
    :param time_granularity: the time granularity of the graphs time steps- can be 'days', 'weeks', 'months', 'years' or 'hours'
    :return:
    '''
    temporal_g = TemporalGraph(data=graph_df, time_granularity=time_granularity, dataset_name=dataset_name)
    graph_df = temporal_g.data
    # graph_df['time'] = graph_df['time_index']
    graph_df.loc[:, 'time'] = graph_df['time_index']
    graph_nx = nx.from_pandas_edgelist(graph_df, 'source', 'target', edge_attr=['time'],
                                       create_using=nx.MultiDiGraph())
    return graph_nx, temporal_g


if __name__ == '__main__':
    # load facebook
    # dataset_name = 'facebook'
    # graph_path = 'data/facebook/facebook-wall.txt'
    # graph_df = pd.read_table(graph_path, sep='\t', header=None)
    # graph_df.columns = ['source', 'target', 'time']
    # graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
    # graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    # print(len(graphs))

    # load weekly enron
    dataset_name = 'enron_m'
    graph_path =  'data/enron/out.enron'
    graph_df = pd.read_table(graph_path, sep=' ', header=None)
    graph_df.columns = ['source', 'target', 'weight', 'time']
    graph_nx, temporal_graph = load_dataset(graph_df, dataset_name, time_granularity='months')
    graphs = temporal_graph.get_temporal_graphs(min_degree=5)
    print(graphs)
    print(len(graphs))
    with open('./data/enron/enron_months.pkl', 'wb') as new_file:
        pickle.dump(graphs, new_file)
    #
    # load game of thrones
    # dataset_name = 'game_of_thrones'
    # with open('./data/game_of_thrones/gameofthrones_2017_graphs_dynamic.pkl', 'rb') as file:
    #     graphs = pickle.load(file)
    #     print(len(graphs))

    # load slashdot
    # dataset_name = 'slashdot'
    # with open('./data/slashdot/slashdot_monthly_dynamic.pkl', 'rb') as file:
    #     graphs = pickle.load(file)
    #     print(len(graphs))

    # #load formula1
    # dataset_name = 'formula'
    # with open('./data/formula/formula_2019_graphs_dynamic.pkl', 'rb') as file:
    #     graphs = pickle.load(f)
