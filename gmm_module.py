# encoding: utf-8
import numpy as np
import random
import math
import os
import pickle
import torch

# 確率を計算するためのクラス
class GaussWishart():
    def __init__(self,dim, mean, var):
        # 事前分布のパラメータ
        self.__dim = dim
        self.__r0 = 1
        self.__nu0 = dim + 2
        self.__m0 = mean.reshape((dim,1))
        self.__S0 = np.eye(dim, dim ) * var

        self.__X = np.zeros( (dim,1) )
        self.__C = np.zeros( (dim, dim) )
        self.__r = self.__r0
        self.__nu = self.__nu0
        self.__N = 0

        self.__update_param()

    def __update_param(self):
        self.__m = (self.__X + self.__r0 * self.__m0)/(self.__r0 + self.__N )
        self.__S = - self.__r * self.__m * self.__m.T + self.__C + self.__S0 + self.__r0 * self.__m0 * self.__m0.T;

    def add_data(self, x):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X += x
        self.__C += x.dot( x.T )
        self.__r += 1
        self.__nu += 1
        self.__N += 1
        self.__update_param()

    def delete_data(self, x):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X -= x
        self.__C -= x.dot( x.T )
        self.__r -= 1
        self.__nu -= 1
        self.__N -= 1
        self.__update_param()

    def calc_loglik(self, x):
        def _calc_loglik(self):
            p = - self.__N * self.__dim * 0.5 * math.log( math.pi )
            p+= - self.__dim * 0.5 * math.log( self.__r )
            p+= - self.__nu * 0.5 * math.log( np.linalg.det( self.__S ) );

            for d in range(1,self.__dim+1):
                p += math.lgamma( 0.5*(self.__nu+1-d) )

            return p

        # log(P(X))
        p1 = _calc_loglik( self )

        # log(P(x,X))
        self.add_data(x)
        p2 = _calc_loglik( self )
        self.delete_data(x)

        # log(P(x|X)) = log(P(x,X)) - log(P(X))
        return p2 - p1

    def get_loglik(self):
         p = - self.__N * self.__dim * 0.5 * math.log( math.pi )
         p+= - self.__dim * 0.5 * math.log( self.__r )
         p+= - self.__nu * 0.5 * math.log( np.linalg.det( self.__S ) );

         for d in range(1,self.__dim+1):
             p += math.lgamma( 0.5*(self.__nu+1-d) )

         return p
    
    def get_mean(self):
        return self.__m

    def get_num_data(self):
        return self.__N

    def get_param(self):
        return [self.__X, self.__C, self.__r, self.__nu, self.__N, self.__m0]
    
    def load_params(self, params):
        self.__X = params[0]
        self.__C = params[1]
        self.__r = params[2]
        self.__nu = params[3]
        self.__N = params[4]
        self.__m0 = params[5]

        self.__update_param()


def calc_probability( dist, d ):
    return dist.get_num_data() * math.exp( dist.calc_loglik( d ) )


def sample_class( d, distributions, i ):
    K = len(distributions)
    P = [ 0.0 ] * K

    # 累積確率を計算
    P[0] = calc_probability( distributions[0], d ) 
    for k in range(1,K):
        P[k] = P[k-1] + calc_probability( distributions[k], d ) 


    # サンプリング
    rnd = P[K-1] * random.random()
    for k in range(K):
        if P[k] >= rnd:
            return k


def calc_acc( results, correct ):
    K = np.max(results)+1     # カテゴリ数
    N = len(results)          # データ数
    max_acc = 0               # 精度の最大値
    changed = True            # 変化したかどうか

    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( N )

                # iとjを入れ替える
                for n in range(N):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # 精度を計算
                acc = (tmp_result==correct).sum()/float(N)

                # 精度が高くなって入れば保存
                if acc > max_acc:
                    max_acc = acc
                    results = tmp_result
                    changed = True

    return max_acc, results

# モデルの保存
def save_model( Pdz, mu, classes, save_dir, categories, distributions, liks, load_dir ):
    if not os.path.exists( save_dir ):
        os.makedirs( save_dir )
    
    if load_dir is None:
        # モデルパラメータの保存
        K = len(Pdz[0])
        params = []
        for k in range(K):
            params.append(distributions[k].get_param())
        with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
            pickle.dump( params, f )
        # muと尤度の保存
        np.savetxt( os.path.join( save_dir, "mu.txt" ), mu )
        np.savetxt( os.path.join( save_dir, "liklihood.txt" ), liks, fmt="%f" )

    # 確率の保存
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt="%f" )
    
    # 分類結果・精度の計算と保存
    if categories is not None:
        acc, results = calc_acc( classes, categories )
        np.savetxt( os.path.join( save_dir, "class.txt" ), results, fmt="%d" )
        np.savetxt( os.path.join( save_dir, "acc.txt" ), [acc], fmt="%f" )
        
    else:
        np.savetxt( os.path.join( save_dir, "class.txt" ), classes, fmt="%d" )

# モデルパラメータの読み込み
def load_model( load_dir, distributions ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open( model_path, "rb" ) as f:
        a = pickle.load( f )
    
    K = len(distributions)
    for k in range(K):
        distributions[k].load_params(a[k])

def test( data, K, categories=None, load_dir=None ):
    # データ数
    N = len(data)
    # データの次元
    dim = len(data[0])
    
    # 尤度のリスト
    liks = []

    Pdz = np.zeros((N,K))
    mu = np.zeros((N,dim))

    # データをランダムに分類
    classes = np.random.randint( K , size=N )

    # ガウス-ウィシャート分布の生成
    mean = np.mean( data, axis=0 )
    distributions = [ GaussWishart(dim, mean , 0.1) for _ in range(K) ]
        
    print("GMM：認識モード")
    load_model(load_dir, distributions)
    
    # 認識
    for n in range(N):
        for k in range(K):
            Pdz[n][k] = calc_probability( distributions[k], data[n] )
    classes = np.argmax(Pdz, axis=1)
    
    # 正規化
    Pdz = (Pdz.T / np.sum(Pdz,1)).T
    
    #save_model( Pdz, mu, classes, save_dir, categories, distributions, liks, load_dir )

    return torch.from_numpy(Pdz), torch.from_numpy(mu), classes


# gmmメイン
def train( data, K, num_itr=100, save_dir="model", categories=None, load_dir=None ):
    # データ数
    N = len(data)
    # データの次元
    dim = len(data[0])
    
    # 尤度のリスト
    liks = []

    Pdz = np.zeros((N,K))
    mu = np.zeros((N,dim))

    # データをランダムに分類
    classes = np.random.randint( K , size=N )

    # ガウス-ウィシャート分布の生成
    mean = np.mean( data, axis=0 )
    distributions = [ GaussWishart(dim, mean , 0.1) for _ in range(K) ]
        
    # 学習モード時はガウス-ウィシャート分布のパラメータを計算
    if load_dir is None:    
        print("GMM：学習モード")
        for i in range(N):
            c = classes[i]
            x = data[i]
            distributions[c].add_data(x)
    # 認識モード時は学習したモデルパラメータを読み込む
    else:
        print("GMM：認識モード")
        load_model(load_dir, distributions)

    # 学習
    print("GMMの学習メソッド")
    if load_dir is None:
        for it in range(num_itr):
            if it == 0 or (it+1) % 10 == 0:
                print(f"GMMの学習:{it+1}回目")
            # メインの処理
            for i in range(N):
                d = data[i]
                k_old = classes[i]  # 現在のクラス
    
                # データをクラスから除きパラメータを更新
                distributions[k_old].delete_data( d )
                classes[i] = -1
    
                # 新たなクラスをサンプリング(*SERKETではbias_dzがあったが削除)
                k_new = sample_class( d, distributions, i)
    
                # サンプリングされたクラスに更新
                classes[i] = k_new
                
                # サンプリングされたクラスのパラメータを更新
                distributions[k_new].add_data( d )
            
            # 尤度の計算
            lik = 0
            for k in range(K):
                lik += distributions[k].get_loglik()
            liks.append( lik )

        for n in range(N):
            for k in range(K):
                Pdz[n][k] = calc_probability( distributions[k], data[n] )
                if classes[n] == k:
                    mu[n] = distributions[k].get_mean().reshape((1,dim))[0]
    
    # 認識
    if load_dir is not None:
        for n in range(N):
            for k in range(K):
                Pdz[n][k] = calc_probability( distributions[k], data[n] )
        classes = np.argmax(Pdz, axis=1)

    # 正規化
    Pdz = (Pdz.T / np.sum(Pdz,1)).T
    
    save_model( Pdz, mu, classes, save_dir, categories, distributions, liks, load_dir )

    return torch.from_numpy(Pdz), torch.from_numpy(mu)

