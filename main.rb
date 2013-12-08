#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require "optparse"
require './nbc.rb'

################################################################################
# default
################################################################################
@ifile = "relation.dat"  # 入力ファイル
@ofile = "result.dat"    # 出力ファイル
@m = 0                   # 学習法
@a = 1.1
@b = 1.1
@k = 5
@r = 1
@@mm, @@em, @@me, @@ee = [0, 1, 2, 3]

################################################################################
# Arguments
################################################################################
OptionParser.new { |opts|
  # options
  opts.on("-h","--help","Show this message") {
    puts opts
    exit
  }
  opts.on("-i [INPUT]"){ |f|
    @ifile = f
  }
  opts.on("-o [OUTPUT]"){ |f|
    @ofile = f
  }
  opts.on("-k [# clasters]"){ |f|
    @k = f.to_i
  }
  opts.on("-r [# restart]"){ |f|
    @r = f.to_i
  }
  opts.on("-a [alpha]"){ |f|
    @a = f.to_f
  }
  opts.on("-b [beta]"){ |f|
    @b = f.to_f
  }
  opts.on("-1", "--mm"){
    @m = @@mm
  }
  opts.on("-2", "--em"){
    @m = @@em
  }
  opts.on("-3", "--me"){
    @m = @@me
  }
  opts.on("-4", "--ee"){
    @m = @@ee
  }
  # parse
  opts.parse!(ARGV)
}

################################################################################
# load data
################################################################################
class MyData
  #### new ####
  attr_reader :d    # データの次元
  attr_reader :n    # n[d] = d次元の属性値の種類
  attr_reader :data # データ
  def initialize(_file)
    # データの読み取り
    # データ形式 : i a_i,1 ... a_i,d
    @raw_data = []
    open(_file).read.split("\n").each do |line|
      ary = line.split(" ")
      name = ary.shift # 名前
      attr = ary       # 属性
      @raw_data.push([name, attr])
    end

    # データの加工
    @N = @raw_data.size                     # データ数
    @d = @raw_data[0][1].size               # データの次元
    @ns = Array.new(@N)                     # 各データ点の名前
    @vs = Array.new(@d){|d| Hash.new(nil)}  # 各次元の値

    @data = []
    for i in 0..@N-1
      name, attr = @raw_data[i]
      # 名前
      @ns[i] = name

      # 属性
      a = Array.new(@d)
      for d in 0..@d-1
        v = attr[d]
        @vs[d][v] = @vs[d].size if @vs[d][v] == nil
        a[d] = @vs[d][v]
      end
      @data.push(a)
    end
    @n = Array.new(@d){|d| @vs[d].size}
  end

  #### clastering ####
  def clastering(_file, opt = {})
    opt = {:m => 0, :a => 1.1, :b => 1.1, :k => 5, :r => 1}.merge(opt)
    p opt
    # init
    m = NaiveBayesModel.new(opt[:k], @n)
    m.a = opt[:a]
    m.b = opt[:b]
    m.restart = opt[:r]
    @data.each{|a| m.add(a) }

    # learn
    m.rand_p
    case opt[:m]
    when @@mm then m.mm_learn
    when @@em then m.em_learn
    when @@me then m.me_learn
    when @@ee then m.ee_learn
    end

    # result
    f = open(_file, "w")
    m.result.sort{|a,b| b.size <=> a.size}.each do |c|
      f.puts "#{c.map{|i| @ns[i]}.join(" ")}"
    end
    f.close
  end
end

################################################################################
# main
################################################################################
d = MyData.new(@ifile)
d.clastering(@ofile, {:m => @m, :a => @a, :b => @b, :k => @k, :r => @r})
