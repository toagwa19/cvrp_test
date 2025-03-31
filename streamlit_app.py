import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import folium
import math
from streamlit.components.v1 import html
from amplify import VariableGenerator
from amplify import one_hot
from amplify import einsum
from amplify import less_equal, ConstraintList
from amplify import Poly, einsum
from amplify import Model
from amplify import FixstarsClient
from datetime import timedelta
from amplify import solve
from geopy.distance import geodesic
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def tsp_dynamic_programming(data):
    route = []
    if len(data) == 1:
      print("拠点０")
      route.append(0)
      return 0, route

    result = []
    distance_matrix = []
    for k, v, d,e in data:
      tmp = []
      for kk, vv, dd,ee in data:
        tmp.append(math.sqrt((k - kk)**2 + (v - vv)**2))
      result.append(tmp)
    distance_matrix =result

    if len(data) > 18:
      print("近似解法：2-Opt法")
      min_distance,route_2opt = tsp_2opt(distance_matrix)
      return min_distance,route_2opt


    n = len(distance_matrix)

    # DPテーブルと経路テーブルの初期化
    dp = [[float('inf')] * (1 << n) for _ in range(n)]
    path = [[-1] * (1 << n) for _ in range(n)]

    # スタート地点の初期化
    dp[0][1] = 0

    # 動的計画法で最短距離を計算
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v) or u == v:
                    continue
                if dp[v][mask | (1 << v)] > dp[u][mask] + distance_matrix[u][v]:
                    dp[v][mask | (1 << v)] = dp[u][mask] + distance_matrix[u][v]
                    path[v][mask | (1 << v)] = u

    # 終点からスタート地点に戻る最短距離を計算
    end_mask = (1 << n) - 1
    min_distance = min(dp[v][end_mask] + distance_matrix[v][0] for v in range(1, n))

    # 最短距離を持つ経路を復元
    last_city = min(range(1, n), key=lambda v: dp[v][end_mask] + distance_matrix[v][0])

    mask = end_mask
    while last_city != -1:
        route.append(last_city)
        prev_city = path[last_city][mask]
        mask ^= (1 << last_city)
        last_city = prev_city

    route.reverse()
    route.append(0)

    return min_distance, route

def calculate_total_distance(route, distance_matrix):
    """経路の総距離を計算"""
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1]][route[0]]

def swap_2opt(route, i, k):
    """2つの辺を入れ替えた新しい経路を作成"""
    new_route = route[0:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route

def tsp_2opt(distance_matrix):
    """2-Opt法による巡回セールスマン問題の近似解法"""
    n = len(distance_matrix)
    #print(distance_matrix)
    # ランダムな初期巡回路を生成
    route = list(range(n))
    random.shuffle(route)

    improvement = True
    best_distance = calculate_total_distance(route, distance_matrix)

    while improvement:
        improvement = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                new_route = swap_2opt(route, i, k)
                new_distance = calculate_total_distance(new_route, distance_matrix)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improvement = True
        # 改善がなくなるまでループ

    return  best_distance,route

def plot_solution(coord: list[tuple], title: str, best_tour: dict = dict()):
    l = len(coord)
    center = [
        np.sum(lat for _, lat in coord) / l,
        np.sum(lon for lon, _ in coord) / l,
    ]
    m = folium.Map(center, tiles="OpenStreetMap", zoom_start=5)
    folium.Marker(
        location=coord[0][::-1],
        popup=f"depot",
        icon=folium.Icon(icon="car", prefix="fa"),
    ).add_to(m)

    _color = _colors[1]
    if best_tour:
        for k, tour in best_tour.items():
            _color = _colors[k % len(_colors)]
            for city in tour:
                if city == 0:
                    continue

                folium.Marker(
                    location=coord[city][::-1],
                    popup=f"person{k}",
                    icon=folium.Icon(
                        icon="user", prefix="fa", color="white", icon_color=_color
                    ),
                ).add_to(m)
            folium.vector_layers.PolyLine(  # type: ignore
                locations=[coord[city][::-1] for city in tour], color=_color, weight=3
            ).add_to(m)
    else:
        for k, node in enumerate(coord):
            if k == 0:
                continue
            folium.Marker(
                location=node[::-1],
                popup=f"customer{k}",
                icon=folium.Icon(
                    icon="user", prefix="fa", color="white", icon_color=_color
                ),
            ).add_to(m)

    title = f"<h4>{title}</h4>"
    m.get_root().html.add_child(folium.Element(title))  # type: ignore

    return m

_colors = [
    "green",
    "orange",
    "blue",
    "red",
    "purple",
    "pink",
    "darkblue",
    "cadetblue",
    "darkred",
    "lightred",
    "darkgreen",
    "lightgreen",
    "lightblue",
    "darkpurple",
    "yellow",
    "darkyellow",
    "black",
    "lightyellow",
]


def upperbound_of_tour(capacity: int, demand: np.ndarray) -> int:
    max_tourable_cities = 0
    for w in sorted(demand):
        capacity -= w
        if capacity >= 0:
            max_tourable_cities += 1
        else:
            return max_tourable_cities
    return max_tourable_cities

def process_sequence(sequence: dict[int, list]) -> dict[int, list]:
    new_seq = dict()
    for k, v in sequence.items():
        v = np.append(v, v[0])
        mask = np.concatenate(([True], np.diff(v) != 0))
        new_seq[k] = v[mask]
    return new_seq

#結果取得・可視化
# one-hot な変数テーブルを辞書に変換。key：車両インデックス, value：各車両が訪問した都市の順番が入ったリスト
def onehot2sequence(solution: np.ndarray) -> dict[int, list]:
    nvehicle = solution.shape[2]
    sequence = dict()
    for k in range(nvehicle):
        sequence[k] = np.where(solution[:, :, k])[1]
    return sequence


def subsolution(nvehicle,capacity,demand,ncity,ind2coord) -> dict[int, list]:
  avg_cities_per_vehicle = ncity // nvehicle

  # 各都市における配送需要（重量）を決定
  demand_max = np.max(demand)
  demand_mean = demand.mean()

  # 座標の取り得る範囲を設定
  lat_range = [0, 70]
  lon_range = [0, 70]

  # 2都市間の座標距離行列 D
  #distance_matrix = np.array(
  #    [
  #        [geodesic(coord_i[::-1], coord_j[::-1]).m for coord_j in ind2coord]
  #        for coord_i in ind2coord
  #    ]
  #)

  #y_distance_matrix = []
  #y_result = []
  #for k, v in ind2coord:
  #    tmp = []
  #    for kk, vv in ind2coord:
  #        tmp.append(math.sqrt((k - kk)**2 + (v - vv)**2))
  #    y_result.append(tmp)
  #y_distance_matrix =y_result
  #print(y_distance_matrix)
  # 距離行列を作成
  y_distance_matrix = np.zeros((len(ind2coord), len(ind2coord)))

  for i, (k, v) in enumerate(ind2coord):
      for j, (kk, vv) in enumerate(ind2coord):
          y_distance_matrix[i, j] = math.sqrt((k - kk)**2 + (v - vv)**2)
  #print(y_distance_matrix)

  gen = VariableGenerator()

  # 積載可能量から1台の車両が訪問できる都市の最大数
  max_tourable_cities = upperbound_of_tour(capacity, demand)

  x = gen.array("Binary", shape=(max_tourable_cities + 2, ncity + 1, nvehicle))

  x[0, 1:, :] = 0
  x[-1, 1:, :] = 0

  x[0, 0, :] = 1
  x[-1, 0, :] = 1

  one_trip_constraints = one_hot(x[1:-1, :, :], axis=1)
  one_visit_constraints = one_hot(x[1:-1, 1:, :], axis=(0, 2))
  weight_sums = einsum("j,ijk->ik", demand, x[1:-1, 1:, :])

  capacity_constraints: ConstraintList = less_equal(
      weight_sums,  # type: ignore
      capacity,
      axis=0,
      penalty_formulation="Relaxation",
  )

  max_tourable_cities = x.shape[0]
  dimension = x.shape[1]
  nvehicle = x.shape[2]

  # 経路の総距離
  objective: Poly = einsum("pq,ipk,iqk->", y_distance_matrix, x[:-1], x[1:])  # type: ignore

  constraints = one_trip_constraints + one_visit_constraints + capacity_constraints
  constraints *= np.max(y_distance_matrix)  # 重みの設定

  model = Model(objective, constraints)

  #クライアントの設定
  client = FixstarsClient()
  client.parameters.timeout = timedelta(milliseconds=2000)  # タイムアウト2秒
  #client.token = "AE/E4ZQ7gM9oXlHVHgFJIHCWfz2LpB0vYaU"
  client.token = "AE/Vu9TgRFhbjSvIbDYr2HLcU5TdzstS1NG"
  #"SE/5VtmyjP9SlweYv2tgHIgSgiJ7zbxUJKy"

  #作成したモデルとクライアントを solve 関数に与えることで求解を行います。

  result = solve(model, client)
  if len(result) == 0:
      #raise RuntimeError("Some of the constraints are not satisfied.")
      print(f"★★★QUBO制約違反:Some of the constraints are not satisfied.")
      return {},9999999999999999

  x_values = result.best.values  # 目的関数の値が最も低い解の変数値の取り出し

  solution = x.evaluate(x_values)  # 結果が入ったnumpy配列
  sequence = onehot2sequence(
      solution
  )  # one-hot な変数テーブルを辞書に変換。key：車両インデックス, value：各車両が訪問した都市の順番が入ったリスト
  best_tour = process_sequence(sequence)  # 上の辞書からデポへの余計な訪問を取り除く
  #print(f"Cost: {result.solutions[0].objective}")  # 目的関数値を表示
  #print(f"All: {result.solutions[0]}")  # 目的関数値を表示
  print(best_tour)
  for root in best_tour:
      tmp=0
      #print(root)
      for rr in best_tour[root]:
          if rr ==0:
              continue
          tmp = tmp+demand[rr-1]
          if tmp > capacity:
              print(f"★★★キャパオーバ制約違反:容量{tmp}、該当ルート{root}、ルート情報{best_tour}")
              return {},9999999999999999
      #print(f"★キャパ:容量{tmp}、該当ルート{root}、ルート情報{best_tour}、需要{demand}")
  #print(*best_tour.items(), sep="\n")  # 求めた解を表示

  #スコア計算
  tmpsum=0
  spoint=0
  epoint=0
  #print(y_distance_matrix)
  #print(ind2coord)
  for i3 in range(nvehicle):
      min_distance=0
      spoint=0
      epoint=0
      for r in range(len(best_tour[i3])):
          if r==len(best_tour[i3])-1:
              break
          spoint=best_tour[i3][r]
          epoint=best_tour[i3][r+1]

          min_distance += y_distance_matrix[spoint][epoint]
      print(f"最短距離: {min_distance}")
      print(f"経路: {best_tour[i3]}")
      tmpsum += min_distance

  print(f"★アニーリング総距離: {tmpsum}")
  #print(f"★歴代ベスト総距離: {totalsum}")

  return best_tour ,tmpsum

















# ★タイトルの表示
st.title("CVRPアプリ")
st.file_uploader("CVRP入力ファイルアップロード", type='csv')
# サンプルの顧客データ（x, y座標）
depo=[35,35,0,0]

m_customers = np.array([
#[35,35],
[22,22],
[36,26],
[21,45],
[45,35],
[55,20],
[33,34],
[50,50],
[55,45],
[26,59],
[40,66],
[55,65],
[35,51],
[62,35],
[62,57],
[62,24],
[21,36],
[33,44],
[9,56],
[62,48],
[66,14],
[44,13],
[26,13],
[11,28],
[7,43],
[17,64],
[41,46],
[55,34],
[35,16],
[52,26],
[43,26],
[31,76],
[22,53],
[26,29],
[50,40],
[55,50],
[54,10],
[60,15],
[47,66],
[30,60],
[30,50],
[12,17],
[15,14],
[16,19],
[21,48],
[50,30],
[51,42],
[50,15],
[48,21],
[12,38],
[37,52],
[49,49],
[52,64],
[20,26],
[40,30],
[21,47],
[17,63],
[31,62],
[52,33],
[51,21],
[42,41],
[31,32],
[5,25],
[12,42],
[36,16],
[52,41],
[27,23],
[17,33],
[13,13],
[57,58],
[62,42],
[42,57],
[16,57],
[8,52],
[7,38],
[27,68],
[30,48],
[43,67],
[58,48],
[58,27],
[37,69],
[38,46],
[46,10],
[61,33],
[62,63],
[63,69],
[32,22],
[45,35],
[59,15],
[5,6],
[10,17],
[21,10],
[5,64],
[30,15],
[39,10],
[32,39],
[25,32],
[25,55],
[48,28],
[56,37],
[41,49],
[35,17],
[55,45],
[55,20],
[15,30],
[25,30],
[20,50],
[10,43],
[55,60],
[30,60],
[20,65],
[50,35],
[30,25],
[15,10],
[30,5],
[10,20],
[5,30],
[20,40],
[15,60],
[45,65],
[45,20],
[45,10],
[55,5],
[65,35],
[65,20],
[45,30],
[35,40],
[41,37],
[64,42],
[40,60],
[31,52],
[35,69],
[53,52],
[65,55],
[63,65],
[2,60],
[20,20],
[5,5],
[60,12],
[40,25],
[42,7],
[24,12],
[23,3],
[11,14],
[6,38],
[2,48],
[8,56],
[13,52],
[6,68],
[47,47],
[49,58],
[27,43],
[37,31],
[57,29],
[63,23],
[53,12],
[32,12],
[36,26],
[21,24],
[17,34],
[12,24],
[24,58],
[27,69],
[15,77],
[62,77],
[49,73],
[67,5],
[56,39],
[37,47],
[37,56],
[57,68],
[47,16],
[44,17],
[46,13],
[49,11],
[49,42],
[53,43],
[61,52],
[57,48],
[56,37],
[55,54],
[15,47],
[14,37],
[11,31],
[16,22],
[4,18],
[28,18],
[26,52],
[26,35],
[31,67],
[15,19],
[22,22],
[18,24],
[26,27],
[25,24],
[22,27],
[25,21],
[19,21],
[20,26],
[18,18],
])

# 各ノードの需要
m_demands = [18,26,11,30,21,19,15,16,29,26,37,16,12,31,8,19,20,13,15,22,28,12,6,27,14,18,17,29,13,22,25,28,27,19,10,12,14,24,16,33,15,11,18,17,21,27,19,20,5,7,30,16,9,21,15,19,23,11,5,19,29,23,21,10,15,3,41,9,28,8,8,16,10,28,7,15,14,6,19,11,12,23,26,17,6,9,15,14,7,27,13,11,16,10,5,25,17,18,10,10,7,13,19,26,3,5,9,16,16,12,19,23,20,8,19,2,12,17,9,11,18,29,3,6,17,16,16,9,21,27,23,11,14,8,5,8,16,31,9,5,5,7,18,16,1,27,36,30,13,10,9,14,18,2,6,7,18,28,3,13,19,10,9,20,25,25,36,6,5,15,25,9,8,18,13,14,3,23,6,26,16,11,7,41,35,26,9,15,3,1,2,22,27,20,11,12,10,9,17]


customers = np.insert(m_customers, 2, m_demands, axis=1)

# デポ（スタート地点）の座標
depot = np.array([35, 35])

# 各車両の容量
vehicle_capacity = 200

# クラスタ数（車両数に対応）
n_clusters = 18
print(depot)
# 顧客の座標からデポへの相対位置（角度）を計算
angles = np.arctan2(customers[:, 1] - depot[1], customers[:, 0] - depot[0])

# 顧客データに角度情報を追加
customers_with_angles = np.hstack([customers, angles.reshape(-1, 1)])

# 角度に基づいて顧客をソート
customers_with_angles = customers_with_angles[np.argsort(customers_with_angles[:, 3])]

# 都市の数
st.write("①拠点表示")
ncity = len(m_demands)
title = f"capacity={vehicle_capacity}, ncity={ncity}, nvehicle={n_clusters}"
ind2coord = np.insert(m_customers, 0,depot, axis=0)
    
# 地図をStreamlit上に表示
#st_folium(plot_solution(ind2coord, title), width=900, height=500, key="unique_key")
# 地図をHTMLに変換
m0=plot_solution(ind2coord, title)
m_html0 = m0._repr_html_()
# ★★★Streamlit で Folium の地図を表示する
html(m_html0, width=1000,height=600)

























# ★ボタンを作成
if st.button('古典解法を実施'):
    # 試行回数
    num = 262
    totalroute=0
    totalsum=9999
    bestlabels=[]
    bestseed=0
    #17 乱数シード: 261 1390
    for seed in range(261,num,1):
      # 角度に基づいて顧客をクラスタリング
      kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
      kmeans.fit(customers_with_angles[:, :2])

      # クラスタリングの結果を取得
      labels = kmeans.labels_

      # 各クラスタの重心をデポも含めて計算し、調整
      centroids = []
      for i in range(n_clusters):
          cluster_points = customers_with_angles[labels == i][:, :2]  # x, y座標のみを使用
          if cluster_points.size > 0:
              # デポを含めた重心を計算
              all_points = np.vstack([depot, cluster_points])
              centroid = np.mean(all_points, axis=0)
              centroids.append(centroid)
          else:
              centroids.append([0, 0])

      centroids = np.array(centroids)

      # デポから各クラスタ重心への距離を考慮して再割り当て
      for i in range(len(customers_with_angles)):
          distances = cdist([customers_with_angles[i, :2]], centroids)
          min_index = np.argmin(distances)
          #max_index = np.argmax(distances)
          labels[i] = min_index

      # 容量制約を考慮した再調整
      r_flg1 = True
      r_flg2 = True
      cnt_flg1 =0
      r_flg3 = True
      r_flg4 = True
      cnt_flg3 =0
      x_roop=1
      y_roop=1
      for cnt in range(100):  # 最大100回の調整
          print(cnt)
          cluster_demands = np.zeros(n_clusters)
          #終了条件１
          if r_flg1 == False and r_flg2 == True:
            #x_roop=min(x_roop+1,5)
            if cnt_flg1 >= 3:
              r_flg2 = False
            cnt_flg1 = cnt_flg1 + 1
          r_flg1 = False
          #終了条件２
          if r_flg3 == False and r_flg4 == True:
            y_roop=min(y_roop+1,5)
            if cnt_flg3 >= 3:
              r_flg4 = False
            cnt_flg3 = cnt_flg3 + 1
          r_flg3 = False

          for i in range(n_clusters):
              cluster_demands[i] = np.sum(customers_with_angles[labels == i][:, 2])

          for i in range(n_clusters):
              if cluster_demands[i] > vehicle_capacity:
                  # 需要が多すぎる場合、需要の高い顧客を他のクラスタに再割り当て
                  print("キャパオーバー１")
                  cluster_indices = np.where(labels == i)[0]
                  demands = customers_with_angles[cluster_indices][:, 2]
                  sorted_indices = np.argsort(demands)[::-1]  # 需要が大きい順にソート

                  tmp_distances = cdist(centroids[i].reshape(1, -1),customers_with_angles[cluster_indices, :2])[0,:]
                  sorted_indices = np.argsort(tmp_distances)[::-1]

                  for idx in sorted_indices:
                      if cluster_demands[i] <= vehicle_capacity:
                          print("キャパオーバー解消")
                          break
                      # 最も近い別のクラスタに再割り当て
                      distances = cdist([customers_with_angles[cluster_indices[idx], :2]] , centroids)
                      distances[0, i] = np.inf  # 現在のクラスタには再割り当てしない
                      new_label = np.argmin(distances)
                      # 配列を昇順にソートしたインデックスを取得
                      sorted_indices2 = np.argsort(distances)

                      # 乱数
                      #ri = random.choice([0,0,0,0,1,1,2])
                      # x番目に小さい値のインデックスを取得

                      for idx2 in range(len(sorted_indices2[0])):
                          if idx2 > cnt:
                              break
                          new_label = sorted_indices2[0,idx2]

                          if cluster_demands[new_label] + demands[idx] <= vehicle_capacity:
                              print("キャパオーバー移行")
                              labels[cluster_indices[idx]] = new_label
                              cluster_demands[i] -= demands[idx]
                              cluster_demands[new_label] += demands[idx]
                              print(cluster_demands)
                              break
                  # 新しい重心を計算
                  cluster_points = customers_with_angles[labels == i][:, :2]
                  if cluster_points.size > 0:
                      centroids[i] = np.mean(np.vstack([depot, cluster_points]), axis=0)



          # 新しい重心を計算
          for i in range(n_clusters):
              cluster_points = customers_with_angles[labels == i][:, :2]
              if cluster_points.size > 0:
                  centroids[i] = np.mean(np.vstack([depot, cluster_points]), axis=0)
          #print(centroids)

      # クラスタリング結果をプロット
      colors = plt.cm.get_cmap('tab20', n_clusters).colors
      for i in range(n_clusters):
          cluster = customers_with_angles[labels == i]
          plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1} (Demand: {np.sum(cluster[:, 2])})')

      # デポをプロット
      plt.scatter(depot[0], depot[1], c='k', marker='x', label='Depot')
      #plt.figure(figsize=(15, 10))
      plt.xlabel('X coordinate')
      plt.ylabel('Y coordinate')
      plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      plt.title('CVRP Clustering with Depot-Adjusted Centroids and Capacity Constraints')
      plt.show()
    
      # ルート 及び スコア計算
      tmpsum=0

      for i in range(n_clusters):
        data = customers_with_angles[labels == i]
        data = np.insert(data, 0,depo, axis=0)

        min_distance, route = tsp_dynamic_programming(data)
        print(f"最短距離: {min_distance}")
        print(f"経路: {' -> '.join(map(str, route))}")
        tmpsum = tmpsum + min_distance

      print(f"乱数シード: {seed}")
      print(f"総距離: {tmpsum}")
      print(f"歴代ベスト総距離: {totalsum}")
      if tmpsum <= totalsum:
        bestseed=seed
        totalsum=tmpsum
        bestlabels=labels
        print(f"記録更新：総距離: {totalsum}")






    # ランダムなデータを生成
    data = np.random.randn(100)

    # データをDataFrameに変換
    df = pd.DataFrame(data, columns=["value"])

    # データの統計情報を表示
    #st.write("データの統計情報:")
    #st.write(df.describe())

    # 折れ線グラフを作成
    #fig, ax = plt.subplots()
    #ax.plot(df.index, df['value'])
    #ax.set_title('Graph')
    #ax.set_xlabel('Index')
    #ax.set_ylabel('Value')
    # グラフを表示
    #st.pyplot(fig)    

    # ルート 及び スコア計算
    tmpsum=0
    route_dict = {}
    for i in range(n_clusters):
      new_route =[]
      new_route.append(0)
      cluster_indices9 = np.where(bestlabels == i)[0] +1
      #print(cluster_indices9)

      data = customers_with_angles[bestlabels == i]
      data = np.insert(data, 0,depo, axis=0)
      #print(data)
      min_distance, route = tsp_dynamic_programming(data)
      st.write(f"最短距離{i}: {min_distance}")
      st.write(f"経路{i}: {' -> '.join(map(str, route))}")
      tmpsum = tmpsum + min_distance
      #print(route)
      for ee in route:
          if ee ==0:
              continue
          new_route.append(cluster_indices9[ee-1])
      new_route.append(0)
      route_dict[i] = new_route  # 要素の追加または更新
    #st.write(route_dict)
    st.write(f"総距離: {tmpsum}")
    st.write(f"歴代ベスト総距離: {totalsum}")
    st.write("②解法結果")
    m1=plot_solution(np.insert(customers_with_angles[:,:2], 0,depot, axis=0), title,route_dict)
    # 地図をHTMLに変換
    m_html1 = m1._repr_html_()
    # ★★★Streamlit で Folium の地図を表示する
    html(m_html1, width=1000,height=600)















# ★ボタンを作成
if st.button('量子解法を実施'):


    #★★量子解法★★
    st.write("③量子クラスタリング解法開始")
    # 試行回数
    start = 261
    num = 1
    totalsum=9999
    #bestlabels=[]
    bestseed=0
    #261
    for seed in range(start,start + num,1):
      st.write(f"乱数シード: {seed}")
      # 角度に基づいて顧客をクラスタリング
      kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
      kmeans.fit(customers_with_angles[:, :2])
      # クラスタリングの結果を取得
      labels = kmeans.labels_
      print(labels)
      print(customers_with_angles)
      # 各クラスタの重心をデポも含めて計算し、調整
      centroids = []
      for i5 in range(n_clusters):
          cluster_points = customers_with_angles[labels == i5][:, :2]  # x, y座標のみを使用
          if cluster_points.size > 0:
              # デポを含めた重心を計算
              all_points = np.vstack([depot, cluster_points])
              centroid = np.mean(all_points, axis=0)
              centroids.append(centroid)
          else:
              centroids.append([0, 0])
      centroids = np.array(centroids)
    
      # デポから各クラスタ重心への距離を考慮して再割り当て
      for i6 in range(len(customers_with_angles)):
          distances = cdist([customers_with_angles[i6, :2]], centroids)
          min_index = np.argmin(distances)
          labels[i6] = min_index
    
      # 容量制約を考慮した再調整
      for cnt0 in range(10):  # 最大10回の調整
          print(cnt0)
          cluster_demands = np.zeros(n_clusters)
    
          for i7 in range(n_clusters):
              cluster_demands[i7] = np.sum(customers_with_angles[labels == i7][:, 2])
          print(cluster_demands)
          for i15 in range(n_clusters):
              if cluster_demands[i15] > vehicle_capacity:
                  # 需要が多すぎる場合、需要の高い顧客を他のクラスタに再割り当て
                  print("キャパオーバー１")
                  cluster_indices = np.where(labels == i15)[0]
                  demands = customers_with_angles[cluster_indices][:, 2]
                  sorted_indices = np.argsort(demands)[::-1]  # 需要が大きい順にソート
    
                  tmp_distances = cdist(centroids[i15].reshape(1, -1),customers_with_angles[cluster_indices, :2])[0,:]
                  sorted_indices = np.argsort(tmp_distances)[::-1]
    
                  for idx in sorted_indices:
                      if cluster_demands[i15] <= vehicle_capacity:
                          print("キャパオーバー解消")
                          break
                      # 最も近い別のクラスタに再割り当て
                      distances = cdist([customers_with_angles[cluster_indices[idx], :2]] , centroids)
                      distances[0, i15] = np.inf  # 現在のクラスタには再割り当てしない
                      new_label = np.argmin(distances)
                      # 配列を昇順にソートしたインデックスを取得
                      sorted_indices2 = np.argsort(distances)
    
                      # 乱数
                      # x番目に小さい値のインデックスを取得
                      for idx2 in range(len(sorted_indices2[0])):
                          if idx2 > cnt0:
                              break
                          new_label = sorted_indices2[0,idx2]
    
                          if cluster_demands[new_label] + demands[idx] <= vehicle_capacity:
                              print("キャパオーバー移行")
                              labels[cluster_indices[idx]] = new_label
                              cluster_demands[i15] -= demands[idx]
                              cluster_demands[new_label] += demands[idx]
                              print(cluster_demands)
                              break
                  # 新しい重心を計算
                  cluster_points = customers_with_angles[labels == i15][:, :2]
                  if cluster_points.size > 0:
                      centroids[i15] = np.mean(np.vstack([depot, cluster_points]), axis=0)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
      # 各クラスタの重心をデポも含めて計算し、調整
      centroids = []
      for i11 in range(n_clusters):
          cluster_points = customers_with_angles[labels == i11][:, :2]  # x, y座標のみを使用
          if cluster_points.size > 0:
              # デポを含めた重心を計算
              all_points = np.vstack([depot, cluster_points])
              centroid = np.mean(all_points, axis=0)
              centroids.append(centroid)
          else:
              centroids.append([0, 0])
    
      centroids = np.array(centroids)
    
    
       # 容量制約を考慮した再調整
    
      local_scope=2
      start_scope=2
      fin_flg=True
      old_distance = np.zeros(n_clusters)
      for cnt in range(10):  # 最大10回の調整
          st.write(f"cnt{cnt}回目")
          if fin_flg==False: #1回更新なければ次の乱数
              print(f"★★★更新なし★★★")
              break
          r_flg1=True
          fin_flg=False
          cluster_demands = np.zeros(n_clusters)
          for i10 in range(n_clusters):
              cluster_demands[i10] = np.sum(customers_with_angles[labels == i10][:, 2])
    
          chk_matrix = np.zeros((n_clusters, n_clusters))
          for i in range(n_clusters):
              print(f"cnt{cnt}、i{i}")
              if r_flg1:
                  # (各拠点) ⇔ (重心)　で最も遠い拠点順にソート
                  motocluster_indices = np.where(labels == i)[0]
                  motodemands = customers_with_angles[motocluster_indices][:, 2]
                  motocustomers = customers_with_angles[motocluster_indices][:, :2]
                  tmp_distances = cdist(centroids[i].reshape(1, -1) ,centroids)[0,:]
                  tmp_distances[i] = np.inf  # 現在のクラスタには再割り当てしない
                  sorted_indices = np.argsort(tmp_distances)
    
                  k_cnt=0
                  for idx in sorted_indices:
                      k_cnt += 1
                      new_label = idx                  #print(f"i経過{i}")
                      if new_label == i:
                          break
                      if  k_cnt > max(math.ceil(n_clusters/3),min(n_clusters,6)): #クラスタ近傍数
                          break
                      elif chk_matrix[i,new_label]==1 or chk_matrix[new_label,i]==1:
                          print(f"スキップ:{i},{new_label}")
                          continue
    
                      print(f"{k_cnt}近傍先")
                      chk_matrix[i,new_label]=1
                      chk_matrix[new_label,i]=1
    
                      sakicluster_indices = np.where(labels == new_label)[0]
                      sakidemands = customers_with_angles[sakicluster_indices][:, 2]
                      sakicustomers = customers_with_angles[sakicluster_indices][:, :2]

                      # 車両数
                      nvehicle = 2
                      # 全体的な需要に合わせ、車両の積載可能量 Q を設定する。
                      capacity = vehicle_capacity
                      # 需要数
                      smdemands = np.concatenate((motodemands , sakidemands))
                      # 都市の数
                      ncity = len(smdemands)
                      ## デポと各都市の座標を決定
                      ind2coord = np.concatenate((motocustomers , sakicustomers), axis=0)
                      ind2coord = np.insert(ind2coord, 0,depot, axis=0)
    
                      #indices
                      smcluster_indices = np.concatenate((motocluster_indices , sakicluster_indices))
    
                      #アニーリング探索
                      best_tour, score = subsolution(nvehicle,capacity,smdemands,ncity,ind2coord)
    
                      #既存のスコア
                      tmpsum=0
                      min_distance=0
                      if old_distance[i] == 0:
                          data = customers_with_angles[labels == i]
                          data = np.insert(data, 0,depo, axis=0)
                          min_distance, route = tsp_dynamic_programming(data)
                          print(f"最短距離: {min_distance}")
                          #print(f"経路: {' -> '.join(map(str, route))}")
                          print(f"経路: {route}")
                          old_distance[i]=min_distance
                      else:
                          print(f"既存経路計算をスキップ{i}、{old_distance}")
                          min_distance=old_distance[i]
    
                      #既存のスコア更新
                      tmpsum = tmpsum + min_distance
    
                      if old_distance[new_label] == 0:
                          data = customers_with_angles[labels == new_label]
                          data = np.insert(data, 0,depo, axis=0)
                          min_distance, route = tsp_dynamic_programming(data)
                          print(f"最短距離: {min_distance}")
                          print(f"経路: {route}")
                          old_distance[new_label]=min_distance
                      else:
                          print(f"既存経路計算をスキップ{new_label}、{old_distance}")
                          min_distance=old_distance[new_label]
    
                      tmpsum = tmpsum + min_distance

                      print(f"★既存の総距離: {tmpsum}")
    
                      #スコア更新判定
                      if score < tmpsum:
                          print(f"★★★アニーリング更新★★★ 削減効果: {tmpsum-score}")
                          print(f"現ラベル:{i}、近傍ラベル{new_label}")
                          print(f"クラスター容量{cluster_demands}")
                          old_distance[i]=0
                          old_distance[new_label]=0
                          fin_flg=True

                          for rr in best_tour[0]:
                              if rr ==0:
                                  continue
                              # x番目に小さい値のインデックスを取得
                              labels[smcluster_indices[rr-1]] = i
    
                          for rr in best_tour[1]:
                              if rr ==0:
                                  continue
                              # x番目に小さい値のインデックスを取得
                              labels[smcluster_indices[rr-1]] = new_label
    
                          cluster_demands[i] = np.sum(customers_with_angles[labels == i][:, 2])
                          cluster_demands[new_label] = np.sum(customers_with_angles[labels == new_label][:, 2])

                          bestlabels=labels
                          motocluster_indices = np.where(labels == i)[0]
                          motodemands = customers_with_angles[motocluster_indices][:, 2]

                          motocustomers = customers_with_angles[motocluster_indices][:, :2]
                          # 新しい重心を計算
                          cluster_points = customers_with_angles[labels == i][:, :2]
                          if cluster_points.size > 0:
                              centroids[i] = np.mean(np.vstack([depot, cluster_points]), axis=0)
                              cluster_points = customers_with_angles[labels == new_label][:, :2]
                          if cluster_points.size > 0:
                              centroids[new_label] = np.mean(np.vstack([depot, cluster_points]), axis=0)
    
                          # ルート 及び スコア計算
                          qasum=0
                          route_dict = {}
                          print(f"★★★スコア集計★★★　クラスタ番号: {i}")
                          print(f"クラスター容量{cluster_demands}")

                          for i1 in range(n_clusters):
                            chk_matrix[i1,i]=0
                            chk_matrix[i,i1]=0
                            chk_matrix[i1,new_label]=0
                            chk_matrix[new_label,i1]=0
                            new_route =[]
                            new_route.append(0)
                            cluster_indices9 = np.where(labels == i1)[0] +1
    
                            data = customers_with_angles[labels == i1]
                            data = np.insert(data, 0,depo, axis=0)
                            min_distance, route = tsp_dynamic_programming(data)
                            print(f"最短距離(ラベル{i}): {min_distance}")
                            #print(f"経路(ラベル{i}): {' -> '.join(map(str, route))}")
                            print(f"経路: {route}")
                            qasum = qasum + min_distance
    
                            old_distance[i1]=min_distance
    
                            for ee3 in route:
                                if ee3 ==0:
                                    continue
                                new_route.append(cluster_indices9[ee3-1])
                            new_route.append(0)
                            route_dict[i1] = new_route  # 要素の追加または更新
                          #print(route_dict)
    
                          chk_matrix[i,new_label]=1
                          chk_matrix[new_label,i]=1
                          print(f"総距離: {qasum}")
                          if totalsum > qasum:
                              st.write(f"③量子クラスタリング解法結果　★★★記録更新★★★総距離: {qasum}")
                              totalsum=qasum
                              bestlabels=labels
                              st.write(f"乱数シード{seed}、{cnt}回目、更新クラスタ{i}番目、{k_cnt}近傍先")
                              m3=plot_solution(np.insert(customers_with_angles[:,:2], 0,depot, axis=0), title,route_dict)
                              # 地図をHTMLに変換
                              m_html3 = m3._repr_html_()
                              # ★★★Streamlit で Folium の地図を表示する
                              html(m_html3, width=1000,height=600)


                          print(f"歴代ベスト総距離: {totalsum}")
    
                      tmpsum=0
    
                  # 新しい重心を計算
                  cluster_points = customers_with_angles[labels == i][:, :2]
                  if cluster_points.size > 0:
                      centroids[i] = np.mean(np.vstack([depot, cluster_points]), axis=0)
    
          # 新しい重心を計算
          for i2 in range(n_clusters):
              cluster_points = customers_with_angles[labels == i2][:, :2]
              if cluster_points.size > 0:
                  centroids[i2] = np.mean(np.vstack([depot, cluster_points]), axis=0)
    
          # ルート 及び スコア計算
          qasum=0
          route_dict = {}
          st.write(f"★★★スコア集計★★★　大ループ回数: {cnt}")
          for i9 in range(n_clusters):
            new_route =[]
            new_route.append(0)
            cluster_indices99 = np.where(labels == i9)[0] +1
    
            data = customers_with_angles[labels == i9]
            data = np.insert(data, 0,depo, axis=0)
            min_distance, route = tsp_dynamic_programming(data)
            st.write(f"最短距離{i9}: {min_distance}")
            st.write(f"経路{i9}: {route}")
            qasum = qasum + min_distance
    
            for ee4 in route:
                if ee4 ==0:
                    continue
                new_route.append(cluster_indices99[ee4-1])
            new_route.append(0)
            route_dict[i] = new_route  # 要素の追加または更新
          #st.wirte(route_dict)
          st.write(f"総距離: {qasum}")
          if totalsum > qasum:
              st.write(f"★★★記録更新★★★: {totalsum}")
              totalsum=qasum
              bestlabels = labels
          st.write(f"歴代ベスト総距離: {totalsum}")
          #plot_solution(np.insert(customers_with_angles[:,:2], 0,depot, axis=0), title,route_dict)
          local_scope =local_scope+1

          st.write("③量子クラスタリング解法結果")
          m2=plot_solution(np.insert(customers_with_angles[:,:2], 0,depot, axis=0), title,route_dict)
          # 地図をHTMLに変換
          m_html2 = m2._repr_html_()
          # ★★★Streamlit で Folium の地図を表示する
          html(m_html2, width=1000,height=600)

