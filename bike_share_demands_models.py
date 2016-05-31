from boto import sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn import preprocessing, linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import seaborn as sb
import datetime


outPath = "/home/stm/PycharmProjects/bike_share/"
train_file = '/home/stm/PycharmProjects/bike_share/bike_train.csv'
test_file = '/home/stm/PycharmProjects/bike_share/bike_test.csv'

train = pd.read_csv(train_file, parse_dates=[0])
test = pd.read_csv(test_file, parse_dates=[0])

X_train, X_test = train_test_split(train, test_size=0.2, random_state=42)



# data exploration

datetime =  train['datetime']
season = train['season']
holiday = train['holiday']
workingday = train['workingday']
weather = train['weather']
temp = train['temp']
atemp = train['atemp']
humidity = train['humidity']
windspeed = train['windspeed']
casual = train['casual']
registered = train['registered']
count = train['count']



def splitDatetime(data):
    temp = pd.DatetimeIndex(data['datetime'])
    data['year'] = temp.year
    data['month'] = temp.month
    data['hour'] = temp.hour
    data['weekday'] = temp.weekday
    #data['dateDays'] = (temp.date - temp.date[0]).astype('timedelta64[D]')
    data['dayofweek'] = temp.dayofweek
    #data['datetime_minus_time'] = data["datetime"].apply(lambda data: datetime.datetime(year=data.year, month=data.month, day=data.day))
    #data.set_index(data["datetime_minus_time"],inplace=True)

    data['Saturday']=0
    data.Saturday[data.dayofweek==5]=1

    data['Sunday']=0
    data.Sunday[data.dayofweek==6]=1

    return data

# def plot(data1, data2):
#     plt.plot(data1, data2)
#     plt.xlabel("date")
#     plt.ylabel("data")
#     plt.show()
#
# def histogram(data):
#     plt.hist(data)
#     plt.xlabel("data")
#     plt.ylabel("frequency")
#     plt.show()


def normalize(train, test):
    norm = preprocessing.Normalizer()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test

def createRidge(alpha=1.0):
    est = Ridge(alpha=1.0)
    return est
def createKNN():
    est = KNeighborsRegressor(n_neighbors=2)
    return est
def createGradientBoostingRegressor():
    est = GradientBoostingRegressor()
    return est
def createRandomForest():
    est = RandomForestRegressor(n_estimators=500)
    return est
def createDecisionTree():
    est = DecisionTreeRegressor()
    return est
def createExtraTree():
    est = ExtraTreesRegressor(n_estimators=700)
    return est
def createSVR():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2],'C': [100, 1000, 10000]}]
    est = GridSearchCV(svm.SVR(), tuned_parameters, scoring='r2', n_jobs=-1)
    return est
def createLinearRegression():
    est = linear_model.LinearRegression()
    return est


def predict(est, train, test, features, target):
    est.fit(train[features], train[target])

    with open(outPath + "submission-LinearRegression.csv", 'wb') as f:
        f.write("datetime,count\n")

        for index, value in enumerate(list(est.predict(test[features]))):
            f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))


def main():

    # please read file on your own path

    train_file = '/home/PycharmProjects/bike_share/bike_train.csv'
    test_file = '/home/PycharmProjects/bike_share/bike_test.csv'

    train = pd.read_csv(train_file, parse_dates=[0])
    test = pd.read_csv(test_file, parse_dates=[0])

    train = splitDatetime(train)
    test = splitDatetime(test)



    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #print train

    byday = train.groupby('dayofweek')



    num_casual =  byday['casual'].sum().reset_index()
    num_registered = byday['registered'].sum().reset_index()


    print "number of casual: \n", num_casual
    print "number of registered: \n",  num_registered

    print "counts on season: \n", train['season'].value_counts()
    print "counts on weather: \n", train['weather'].value_counts()


    #compute correlation between features
    feature_cols = ['hour', 'month', 'temp', 'atemp', 'humidity', 'windspeed', 'count', 'registered', 'casual']
    trainFI = train[feature_cols]

    #print "correlation:", (trainFI.corr())

    plt.figure()
    plt.matshow(trainFI.corr())
    plt.colorbar()
    plt.savefig("corrMatrix.png")

    #compute pearson correlation for highly correlated features

    corrPtemp = pearsonr(train.temp, train.atemp)
    #print"correlation_temp_atemp:", (corrPtemp)

    corrPWH = pearsonr(train.weather, train.humidity)
    #print"correlation_weather_humidity", (corrPWH)


    # explaratory analysis

    # sb.set(style="ticks", color_codes=True)
    # sb_plot = sb.pairplot(train, hue="workingday", palette="husl",x_vars=['hour', 'month', 'atemp', 'temp', 'humidity', 'windspeed'], y_vars=["count", "registered","casual"])
    # sb_plot.show()
    # sb_plot = sb.pairplot(train, hue="workingday", palette="husl",y_vars=['atemp', 'humidity', 'windspeed'], x_vars=["month"])
    # sb_plot.show()


    # models
    target = 'count'
    features = [col for col in train.columns if col not in ['datetime', 'casual', 'registered', 'count']]

    #est = createDecisionTree()
    #est = createRandomForest()
    #est = createExtraTree()
    est = createGradientBoostingRegressor()
    #est = createKNN()
    #est = Ridge()
    #est =  createSVR()
    #est =  createLinearRegression()

    predict(est, train, test, features, target)


if __name__ == "__main__":
    main()


    # #X_train, X_cv, y_train, y_cv = train_test_split(train[train_features], train_target, test_size=0.2, random_state=0)
    # #train = splitDatetime(train)
    # #test = splitDatetime(test)
    #
    # X_train = []
    # X_cv = []
    # y_train = []
    # y_cv =[]
    #
    #
    # kf = KFold(len(train_features), n_folds=5)
    #
    # train_features = np.array(train[train_features])
    # train_target = np.array(train_target)
    #
    # print len(train_target)
    # print len(train_features)
    #
    #
    # for train_index, test_index in kf:
    #     X_train_, X_cv_ = train_features[train_index], train_features[test_index]
    #     y_train_,y_cv_ = train_target[train_index], train_target[test_index]
    #
    #     X_train.append(np.array(X_train_))
    #     X_cv.append(np.array(X_cv_))
    #     y_train.append(np.array(y_train_))
    #     y_cv.append(np.array(y_cv_))
    #
    # print y_train
