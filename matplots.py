import matplotlib.pyplot as plt
import numpy as np
import data_handler as dh
from log_printer import get_city_name
import os

dims = ['64', '128', '256']
city_paths = dh.load_city_paths()
cities = ['oldenburg', 'san_francisco', 'anchorage', 'knox_county', 'el_dorado']
categories = ['TSGE1', 'TSGE2', 'TSGE3']


def get_time_values(cityf, dimf):
    times1 = [city_paths[cityf][dimf]['TSGE1']['time'][0], city_paths[cityf][dimf]['TSGE2']['time'][0], city_paths[cityf][dimf]['TSGE3']['time'][0]]
    times2 = [city_paths[cityf][dimf]['TSGE1']['time'][1], city_paths[cityf][dimf]['TSGE2']['time'][1], city_paths[cityf][dimf]['TSGE3']['time'][1]]
    return times1, times2


for city in cities:
    for dim in dims:
        values1, values2 = get_time_values(city, dim)

        bar_width = 0.35
        bar_distance = 0.2
        category_distance = 1.0

        positions1 = np.arange(len(categories)) * (bar_width * 2 + category_distance)
        positions2 = positions1 + bar_width + bar_distance

        fig, ax = plt.subplots()
        bars1 = ax.bar(positions1, values1, bar_width, label=f'Traj Num: {city_paths[cities[0]]["trajNum"][0]}')
        bars2 = ax.bar(positions2, values2, bar_width, label=f'Traj Num: {city_paths[cities[0]]["trajNum"][1]}')

        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Time (sec)')
        ax.set_title(f'{get_city_name(city)}: Dimension {dim}')
        ax.set_xticks(positions1 + (bar_width + bar_distance) / 2)
        ax.set_xticklabels(categories)
        ax.legend()

        output_directory = f'./plots/run_time/{city}/'
        output_filename = f'{city}_{dim}.png'
        output_path = output_directory + output_filename

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(output_path)

        plt.close()

        print(f'Plot saved to {output_path}')

        # plt.show()
