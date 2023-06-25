# 本机测试
https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll

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

## 公式里的{{报错
例如：Liquid syntax error (line 179): Variable '{{y_t}' was not properly terminated with regexp: \}\}/ (Liquid::SyntaxError)

解法，在每篇文章的front matter里面配置render_with_liquid: false。（需要jekyll4，gem install jekyll -v 4.3.2）
因为github-pages-228包依赖jekyll = 3.9.3，所以不能用remote模式

如果需要liquid解析变量，用{% raw %} ... {% endraw %}把公式包起来

https://talk.jekyllrb.com/t/jekyll-liquid-problem-with-a-mathjax-expression/7847/5

https://jekyllrb.com/docs/front-matter/

## 语法问题

### 竖线的错误
Note that LaTeX code that uses the pipe symbol | in inline math statements may lead to a line being recognized as a table line. This problem can be avoided by using the 杠vert command instead of |

### {x}_1报错，改成x_1就ok了。。

### *用杠ast

### 有杠split的部分，整体公式用前后各两个dollar包起来

## mathjax

各种配置项：https://docs.mathjax.org/en/v2.7-latest/config-files.html#common-configurations


# 参考其他人的网站

https://github.com/kevinfossez/kevinfossez.github.io

