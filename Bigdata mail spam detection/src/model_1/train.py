from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql import Row
from pyspark.sql.functions import length

from pyspark.sql.types import *

from sklearn.linear_model import SGDClassifier
import numpy as np
import json
import pickle

from pyspark.ml.feature import Word2Vec,Tokenizer,StopWordsRemover,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline

pkl_filename = "/home/pes1ug19cs334/models/model_1/sgd_2.pkl"
# with open(pkl_filename, 'rb') as file:
#     model = pickle.load(file)
model = SGDClassifier()


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)

#---Initail Pre-processing-----
# tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
# stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
# count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
# idf = IDF(inputCol="c_vec", outputCol="tf_idf")
# ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
# clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
#--------------------------------

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
# word2Vec = Word2Vec(vectorSize=32, seed=42, inputCol="stop_tokens", outputCol="feature")
# word2Vec = Word2Vec(vectorSize=64, seed=42, inputCol="stop_tokens", outputCol="feature")
word2Vec = Word2Vec(vectorSize=128, seed=42, inputCol="stop_tokens", outputCol="feature")
ham_spam_to_num = StringIndexer(inputCol='Class',outputCol='label')
# pca = PCA(k=2, inputCol="feature", outputCol="features")
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,word2Vec])

def do_stuff(x):
    if not x.isEmpty():
        # print("Training")
        temp = []
        for i in x.collect()[0]:
            temp.append(Row(text=i[1],Class=i[2]))
        try:
            sch = StructType([StructField("text", StringType(), True),StructField("Class", StringType(), True)])
            df = spark.createDataFrame(temp, sch)
            df = df.na.drop(how="any")
            data = data_prep_pipe.fit(df)
            data = data.transform(df)
            X = np.array(data.select('feature').collect())
            X = X.squeeze()
            y = np.array(data.select('label').collect())
            y = y.ravel()
            model.partial_fit(X,y, classes = [0.0,1.0])
            print(model.score(X,y))
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
        except Exception as e:
            print("Error here",e)
            pass



batch_data = ssc.socketTextStream("localhost", 6100)
batch_data = batch_data.map(lambda x: json.loads(x))

batch_data.foreachRDD(do_stuff)

ssc.start()
ssc.awaitTermination()
