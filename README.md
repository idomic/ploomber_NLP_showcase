# ploomber_NLP_showcase

Currently in beta
download the model here:
https://drive.google.com/file/d/1Ck_jnx1Z0RX67DUwTbCWJtvkNlxX2Lkt/view?usp=sharing

place it in the folder of the model `checkpoint-80344`

and run the server:
`gunicorn app:app -c ./gunicorn.conf.py -k uvicorn.workers.UvicornWorker`
