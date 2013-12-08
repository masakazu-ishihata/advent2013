#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require './samplers.rb'

################################################################################
# 便利なやつら
################################################################################
#### Array まわり ####
class Array
  def product; inject(:*); end
  def sum; inject(:+); end
  def normalize
    s = self.sum
    s == 0.0 || s == 1.0 ? self.clone : map{|i| i / s.to_f}
  end
end

#### 各種関数 ####
def exp(x); Math.exp(x); end
def log(x); Math.log(x); end
def delta(x, y); x == y ? 1 : 0; end

# digamma 関数 (テイラー近似)
# copied from https://github.com/csauper/content-attitude
def dig(x)
  x = x + 6
  p = 1.0 / (x * x)
  return (((0.004166666666667*p-0.003968253986254)*p+
           0.008333333333333)*p-0.083333333333333)*p+
    Math.log(x) - 0.5 / x - 1.0 / (x-1) - 1.0 / (x-2) -
    1.0 / (x-3) - 1.0 / (x-4) - 1.0 / (x-5) - 1.0 / (x-6)
end

#### 収束判定用 : 平均二乗誤差 ####
def mse(_a, _b)
  e = 0
  for i in 0.._a.size-1
    e += (_a[i] - _b[i]) ** 2
  end
  e
end

################################################################################
# Naive Bayes Model
################################################################################
class NaiveBayesModel
  #### for each instacne / class / dimension / value ####
  def each_i;     for i in 0..@x.size-1; yield i; end; end
  def each_k;     for k in 0..@k-1;      yield k; end; end
  def each_d;     for d in 0..@d-1;      yield d; end; end
  def each_v(_d); for v in 0..@n[_d]-1;  yield v; end; end

  ########################################
  # 初期化
  ########################################
  #### new ####
  attr_reader   :d, :n, :k     # モデル定数
  attr_accessor :a, :b         # ハイパーパラメータ
  attr_reader   :pk, :pvgk     # パラメータ
  attr_reader   :x, :z         # 確率変数
  attr_accessor :restart, :iter, :radius # 学習設定
  def initialize(_k, _n)
    # モデル定数
    @d = _n.size  # 観測ベクトルの次元数
    @n = _n.clone # 各次元の取りうる値
    @k = _k       # クラスタ数

    # ハイパーパラメータ
    @a = 1.0
    @b = 1.0

    # パラメータ
    # @pk[k] = p(k)
    # @pvgk[k][d][v] = p(a_d = v | k)
    @pk   = Array.new(@k){|k| 1 / @k.to_f}
    @pvgk = Array.new(@k){|k| Array.new(@d){|d| Array.new(@n[d]){|v| 1/@n[d].to_f}}}

    # 確率変数
    @x = [] # 観測変数集合 [ a_i = [a_i,1,...,a_i,@d],... ]
    @z = [] # 潜在変数集合 [ k_i,...]

    # 学習用期待値 (もしくはカウント)
    @ek   = Array.new(@k){|k| @a-1}
    @evgk = Array.new(@k){|k| Array.new(@d){|d| Array.new(@n[d]){|v| @b-1}}}

    # 学習設定
    @restart = 1        # ランダムリスタート
    @iter    = 100      # 繰り返し数
    @radius  = 1e-5     # 収束半径
  end

  #### パラメータの初期化 ####
  # rand_p ランダムに初期化
  # max_p  最頻値に初期化
  def rand_p
    @pk = Dirichlet.new_simple(@k, @a).sample
    each_k{|k| each_d{|d| @pvgk[k][d] = Dirichlet.new_simple(@n[d], @b).sample } }
  end
  def max_p
    @pk = @ek.normalize
    each_k{|k| each_d{|d| @pvgk[k][d] = @evgk[k][d].normalize } }
  end

  #### 潜在変数の初期化 ####
  # rand_z ランダムに初期化
  # max_z  最頻値に初期化
  def rand_z; each_i{|i| @z[i] = Categorical.new(@pk).sample}; end
  def max_z;  each_i{|i| @z[i] = max_k(@x[i])}; end
  def max_k(_a)
    mk, mp = [nil, nil]
    each_k{|k| mk, mp = [k, pkga(k, _a)] if mk == nil || mp < pkga(k, _a) }
    mk
  end

  #### 期待値の初期化 ####
  # ハイパーパラメータを考慮した擬似カウント
  def init_e
    each_k{|k| @ek[k] = @a-1}
    each_k{|k| each_d{|d| each_v(d){|v| @evgk[k][d][v] = @b-1}}}
  end

  ########################################
  # データ生成、入力
  ########################################
  #### sample ####
  def sample; sample_with_class[1]; end
  def sample_with_class
    k = Categorical.new(@pk).sample
    a = Array.new(@d){|d| Categorical.new(@pvgk[k][d]).sample }
    [k, a]
  end
  #### add ####
  def add(_a); add_with_class(-1, _a); end
  def add_with_class(_k, _a); @z.push(_k); @x.push(_a); end

  ########################################
  # 確率値 (場合によっては期待値)
  ########################################
  # pka   : p(_k, _a)
  # pkga  : p(_k | _a)
  def pka(_k, _a);  @pk[_k] * pagk(_a, _k); end
  def pkga(_k, _a); pka(_k, _a) / pa(_a);  end

  # pa    : p(_a)
  # pagk  : p(_a | _k)
  def pa(_a);       Array.new(@k){|k| pka(k, _a) }.sum;              end
  def pagk(_a, _k); Array.new(@d){|d| @pvgk[_k][d][_a[d]] }.product; end

  # export/import parameters
  def export_par; [@pk, @pvgk].flatten; end
  def import_par(_ary)
    # load pk
    each_k do |k|
      @pk[k] = _ary.shift
    end
    # load pvgk
    each_k do |k|
      each_d do |d|
        each_v(d) do |v|
          @pvgk[k][d][v] = _ary.shift
        end
      end
    end
  end

  ########################################
  # 学習 : z, p をそれぞれ e, m で推定
  ########################################
  #### z の推定 ####
  # zm_step : z を最頻値で推定
  # ze_step : z を期待値で推定
  def zm_step; max_z; z_step( lambda{|i, k| delta(k, @z[i])} ); end
  def ze_step;        z_step( lambda{|i, k| pkga(k, @x[i])}  ); end
  def z_step(_f)
    init_e
    each_i do |i|
      each_k do |k|
        # ek
        @ek[k] += _f.call(i, k)
        # evgk
        each_d do |d|
          @evgk[k][d][ @x[i][d] ] += _f.call(i, k)
        end
      end
    end
  end

  #### p の推定 ####
  # pm_step : p を最頻値で推定
  # pe_step : p を期待値で推定
  def pm_step; p_step( lambda{|x| log(x)} ); end
  def pe_step; p_step( lambda{|x| dig(x)} ); end
  def p_step(_f)
    sum_ek = @ek.sum
    each_k do |k|
      # pk
      @pk[k] = exp( _f.call(@ek[k]) - _f.call(sum_ek) )
      # pvgk
      each_d do |d|
        sum_evgk = @evgk[k][d].sum
        each_v(d) do |v|
          @pvgk[k][d][v] = exp( _f.call(@evgk[k][d][v]) - _f.call(sum_evgk) )
        end
      end
    end
  end

  #### 学習本体 ####
  def mm_learn; puts "MM"; learn{zm_step; pm_step}; end
  def em_learn; puts "EM"; learn{ze_step; pm_step}; end
  def me_learn; puts "ME"; learn{zm_step; pe_step}; end
  def ee_learn; puts "EE"; learn{ze_step; pe_step}; end
  def learn
    best_p, best_l = [nil, nil]
    for r in 1..@restart
      puts "Try #{r}"

      # パラメータの初期化
      rand_p
      p0 = export_par

      # _n 回 | 収束するまで繰り返す
      for i in 1..@iter
        # learn
        t1 = Time.now
        yield
        t2 = Time.now

        # 収束判定
        p1 = export_par
        e = mse(p0, p1)
        printf "%5d %.5e %.5e\n", i, e, t2-t1
        break if e < @radius
        p0 = p1
      end

      # prediction
      max_p
      max_z

      # loglikelihood
      l = loglikelihood
      printf "LL = %e\n", l

      best_p, best_l = [p1, l] if best_l == nil || best_l < l
    end

    # best result
    import_par(best_p)
    max_z
  end

  #### log likelihood ####
  def loglikelihood
    ll = 0
    each_i{|i| ll += log( pa( @x[i] ) )}
    ll
  end
  def loglikelihood_with_class
    ll = 0
    each_i{|i| ll += log( pagk(@x[i], @z[i]) ) }
    ll
  end

  #### clastering result ####
  def result
    cs = Array.new(@k){|k| []}
    each_i{|i| cs[ @z[i] ].push(i) }
    cs
  end

  ########################################
  # show
  ########################################
  def show
    show_cons
    show_hpar
    show_par
    show_var
    show_exp
  end
  #### constants ####
  def show_cons
    puts "Model Constants"
    puts "D = #{@d}, N = #{@n}, K = #{@k}"
  end
  #### hyper parameters ####
  def show_hpar
    puts "Hyper Parameters"
    puts "a = #{@a}"
    each_d{|d| puts "b = #{@b}"}
  end
  #### parameters ####
  def show_par
    puts "Parameters"
    puts "p(k) = #{@pk}"
    each_k do |k|
      each_d do |d|
        puts "p(a_#{d} | k = #{k}) = #{@pvgk[k][d]}"
      end
    end
  end
  #### variables ####
  def show_var
    puts "Variables"
    each_i do |i|
      puts "#{@z[i]}, #{@x[i]}"
    end
  end
  #### expectations ####
  def show_exp
    puts "Expectations"
    puts "ek = #{@ek}"
    each_k do |k|
      each_d do |d|
        puts "ea[#{k}][#{d}] = #{@evgk[k][d]}"
      end
    end
  end
end
