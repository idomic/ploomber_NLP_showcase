# Ploomber_NLP_showcase

run `pip install ploomber && ploomber install` to get the project dependencies into an isolated environment.

Once done you can activate the virtualenv and run the pipeline:

`source venv-ploomber_NLP_showcase/bin/activate`

`ploomber build`

Currently in beta
download the model here:
https://drive.google.com/file/d/1Ck_jnx1Z0RX67DUwTbCWJtvkNlxX2Lkt/view?usp=sharing

place it in the folder of the model `checkpoint-80344`

and run the server:
`gunicorn app:app -c ./gunicorn.conf.py -k uvicorn.workers.UvicornWorker`


clone the webapp and run `yarn install && yarn start`:
https://github.com/franz101/reactHN

# This pipeline was automatically generated

## Setup

```sh
pip install -r requirements.txt
```

## Usage

List tasks:

```sh
ploomber status
```

Execute:

```sh
ploomber build
```

Plot:

```sh
ploomber plot
```

*Note:* plotting requires `pygraphviz` for instructions, [see this.](https://docs.ploomber.io/en/latest/user-guide/faq_index.html#plotting-a-pipeline)

## Resources

* [Ploomber documentation](https://docs.ploomber.io)
# This pipeline was automatically generated

## Setup

```sh
pip install -r requirements.txt
```

## Usage

List tasks:

```sh
ploomber status
```

Execute:

```sh
ploomber build
```

Plot:

```sh
ploomber plot
```

*Note:* plotting requires `pygraphviz` for instructions, [see this.](https://docs.ploomber.io/en/latest/user-guide/faq_index.html#plotting-a-pipeline)

## Resources

* [Ploomber documentation](https://docs.ploomber.io)