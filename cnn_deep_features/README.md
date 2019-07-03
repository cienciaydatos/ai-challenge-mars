### Create the virtual Environment & Activate the Environment

Note: replace [projectname] for your project name

```
$ python -m venv [projectname]
$ source [projectname]/bin/activate
```

### Install the libraries

Note: replace [path-to-the-file] with the path for your requirements.txt file

```
pip install -r [path-to-the-file]/requirements.txt
```

### Create Ipython Kernel

Note: replace [projectname] for your project name

With the 'venv' activate, run:

```
(venv) $ pip install ipykernel
(venv) $ ipython kernel install --user --name=[projectname]
```
### run your Jupyter Notebook

```
(venv) $ jupyter notebook
```

Select the kernel that you installed previously with the [projectname]
