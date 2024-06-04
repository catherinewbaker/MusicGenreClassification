from initialPreprocessing import gen_Train_and_Test, top_tracks, top_echonest_tracks, top_n_genre_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

from track_name import process_track_names, vectorise

if __name__ == "__main__":                                                       
    sample = top_tracks(True)
    # sample = top_n_genre_tracks(2)
    # sample = top_echonest_tracks()


    # procesed_track_names = process_track_names(sample, True, False, True)
    # tf_idf_track_names = vectorise(procesed_track_names,"tfidf")

    # # With vectorised track names
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,tf_idf_track_names,['track_duration','track_listens','track_favorites'])

    # Just with the three single track features with full coverage
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,['track_duration','track_listens','track_favorites'])

    # Add the recorded date info in
    # sample = top_tracks(daterecorded=True)
    # Combined simple single features
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,['track_duration','track_listens','track_favorites','days_since_first'])

    # adding in track names for smaller sample
    # procesed_track_names = process_track_names(sample, True, False, True)
    # tf_idf_track_names = vectorise(procesed_track_names,"tfidf")
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,tf_idf_track_names,['track_duration','track_listens','track_favorites','days_since_first'])

    # just the records that have the date created attribute, but without using that feature
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,['track_duration','track_listens','track_favorites'])

    print("\n\nTesting for Multi Features")
    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test,"",1)
    # knn(X_train, X_test, y_train, y_test,"",3)
    # knn(X_train, X_test, y_train, y_test,"",5)
    # knn(X_train, X_test, y_train, y_test,"",7)
    # knn(X_train, X_test, y_train, y_test,"",9)
    # knn(X_train, X_test, y_train, y_test,"",21)
    # knn(X_train, X_test, y_train, y_test,"",99)
    # knn(X_train, X_test, y_train, y_test,"",199)
    sgd(X_train, X_test, y_train, y_test)