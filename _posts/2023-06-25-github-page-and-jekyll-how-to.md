---
title:  "使用github page和jekyll搭博客的记录"
date:   2023-06-25 15:17:16 +0800
tags:
  - 教程
---

# theme
用的github排名第一的minimal-mistakes，https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/

# tags,category
tags可以用，category不建议用，加了会改变文章链接（以category为前缀，如果改类目了不方便）

# 本机测试
https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll

新增了gem包，需要执行下面命令
```bash
bundle install
```

本地启动服务器，修改_config文件后要重新启动才能加载变更
```bash
jekyll new --skip-bundle .
```

# 公式支持
可以参考：https://stackoverflow.com/questions/26275645/how-to-support-latex-in-github-pages

theme不同，方法略有不同

## 公式里的{\{报错
例如：Liquid syntax error (line 179): Variable '\{\{y_t}' was not properly terminated with regexp: \}\}/ (Liquid::SyntaxError)

解法：
* 在两个{\{中间加一个空格
* 用{% raw %} ... {% endraw %}把公式包起来

https://talk.jekyllrb.com/t/jekyll-liquid-problem-with-a-mathjax-expression/7847/5

https://jekyllrb.com/docs/front-matter/

## 语法问题

* 竖线改成\vert。Note that LaTeX code that uses the pipe symbol | in inline math statements may lead to a line being recognized as a table line. This problem can be avoided by using the \vert command instead of |
* {x}_1报错，改成x_1就ok了。。
* *用\ast
* 有\split的部分，整体公式用前后各两个dollar包起来
* 公式#号说明：行内公式：\\#，独立公式\#

## mathjax

各种配置项：https://docs.mathjax.org/en/v2.7-latest/config-files.html#common-configurations


# 可供参考的网站

https://github.com/kevinfossez/kevinfossez.github.io

