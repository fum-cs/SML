![](lectures/img/banner.png)

## Computer Science Dept., Ferdowsi University of Mashhad

# Statistical Machine Learning

- [Course Jupyter Book](https://fum-cs.github.io/SML/README.html)

2026 Instructor: Mahmood Amintoosi

---

I should mention that some material was from [Jake VanderPlas's DS book github](https://github.com/jakevdp/PythonDataScienceHandbook/). I have modified his contents to suit my own needs and preferences. I would like to thank him for his great work and generosity.

## Build

In notebooks folder:
- jupyter-book build ./
- copy ..\require.js _build\html\
- copy *.jpg _build\html\_images\
- ghp-import -n -p -f ./_build/html
- jupyter-book build --builder pdflatex ./
