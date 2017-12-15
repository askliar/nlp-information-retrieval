from nltk import SnowballStemmer
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
stop.remove('no')
stop.update(['\'s', ',', '!', '.', '-', ':', '_', ''])

stemmer = SnowballStemmer("english")