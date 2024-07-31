import numpy as np
import pandas as pd
from geopy.distance import distance
from geopy.point import Point
# Haversine 공식을 사용하여 거리 계산 함수 정의
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구의 반지름 (미터 단위)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c  # 미터 단위 거리
    return distance
def generate_random_locations(academy_location, num_samples=10, min_distance_km=1, max_distance_km=24):
    random_locations = []
    for _ in range(num_samples):
        distance_km = np.random.uniform(min_distance_km, max_distance_km)
        angle = np.random.uniform(0, 360)
        origin = Point(academy_location[0], academy_location[1])
        destination = distance(kilometers=distance_km).destination(origin, angle)
        random_locations.append((destination.latitude, destination.longitude))
    return random_locations
