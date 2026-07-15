![](lectures/img/banner.png)

## Computer Science Dept., Ferdowsi University of Mashhad

# Statistical Machine Learning

- [Course Jupyter Book](https://fum-cs.github.io/SML/README.html)

2026 Instructor: Mahmood Amintoosi

## Build

In notebooks folder:
- jupyter-book build ./
- copy ..\require.js _build\html\
- copy *.jpg _build/html/_images (notebooks folder and img folder)
- ghp-import -n -p -f ./_build/html
- jupyter-book build --builder pdflatex ./
- python fix_jupyterbook.py
