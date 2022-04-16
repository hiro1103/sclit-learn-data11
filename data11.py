from cProfile import label
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# データ点の個数：n_samples
# 特徴量の個数：n_features
# クラスタの個数：centers
# クラスタ内の標準偏差：cluster_std
# データ点をシャッフル：shuffle
# 乱数生成器の状態を指定

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)


plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()
# クラスタの個数
# セントロイドの初期値をランダムに選択
# 異なるセントロイドの初期値を用いたk-meansアルゴリズムの実行回数
# k-meansアルゴリズム内部の最大イテレション回数
# 収束と判定するための相対的な許容誤差
# セントロイドの初期化に用いる乱数生成器の状態
# クラスタ中心の計算と各データ点のインデックスの予測
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)


plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

print('Distortion: %.2f' % km.inertia_)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
# シルエット係数を計算
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouseette_vals = silhouette_vals[y_km == c]
    c_silhouseette_vals.sort()
    y_ax_upper += len(c_silhouseette_vals)
    # 色の値をセット
    color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouseette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    # クラスタラベルの表示位置を追加
    yticks.append(y_ax_lower + y_ax_upper / 2.0)
    # 底辺に値に棒の幅を追加
    y_ax_lower += len(c_silhouseette_vals)

# シルエット係数の平均値
silhouette_avg = np.mean(silhouette_vals)
# 係数の平均値に破線を引く
plt.axvline(silhouette_avg, color="red", linestyle="--")
# クラスタラベルを表示
plt.yticks(yticks, cluster_labels+1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolors='black',
            marker='s',
            label='Cluster 1')

plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolors='black',
            marker='o',
            label='Cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Centroids')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
