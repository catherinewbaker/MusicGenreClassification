from initialPreprocessing import gen_Train_and_Test, top_tracks, top_echonest_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

from track_name import process_track_names, vectorise

if __name__ == "__main__":                                                       
    sample = top_echonest_tracks()

    single_features = ['echonest_acousticness','echonest_danceability','echonest_energy','echonest_instrumentalness','echonest_liveness','echonest_speechiness','echonest_tempo','echonest_valence']
    
    # # single feature models for all the echonest features
    # for feature in single_features:
    #     print("\n\nTesting for Single Feature - "+feature)
    #     X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,feature,0)
    #     sgd(X_train, X_test, y_train, y_test)
    #     # nb(X_train, X_test, y_train, y_test)
    #     # knn(X_train, X_test, y_train, y_test,"",-1)

    # combined feature models for all the echonest features
    # print("\n\nTesting for all Echonest Features")
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,single_features)
    # # nb(X_train, X_test, y_train, y_test)
    # # knn(X_train, X_test, y_train, y_test,"",-1)
    # sgd(X_train,X_test,y_train,y_test)

    
  # combined echonest and other simple features
    # print("\n\nTesting for all Echonest Features + simple features")
    # single_echo_plus_simple = single_features + ['track_duration','track_listens','track_favorites']
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,single_echo_plus_simple)
    # # nb(X_train, X_test, y_train, y_test)
    # # knn(X_train, X_test, y_train, y_test,"",-1)
    # sgd(X_train,X_test,y_train,y_test)


    # # combined echonest with trackname
    procesed_track_names = process_track_names(sample, True, False, True)
    tf_idf_track_names = vectorise(procesed_track_names,"tfidf")

    # # just echonest + name
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,tf_idf_track_names,single_features)
    sgd(X_train,X_test,y_train,y_test)


    # # echonest + simple + name
    # single_echo_plus_simple = single_features + ['track_duration','track_listens','track_favorites']
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,tf_idf_track_names,single_echo_plus_simple)

    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test,"",-1)

    # combined echonest plus date recorded
    # sample = top_echonest_tracks(True)
    # print("\n\nTesting for all Echonest Features + date recorded")
    # single_echo_plus_date = single_features + ['days_since_first']
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,single_echo_plus_date)
    # # nb(X_train, X_test, y_train, y_test)
    # # knn(X_train, X_test, y_train, y_test,"",-1)
    # sgd(X_train,X_test,y_train,y_test)
    # # # svm(X_train,X_test,y_train,y_test)
   

    # # combined echonest plus date recorded plus simple
    # sample = top_echonest_tracks(True)
    # print("\n\nTesting for all Echonest Features + date recorded")
    # single_echo_plus_date_and_simple = single_features + ['track_duration','track_listens','track_favorites','days_since_first']
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,single_echo_plus_date_and_simple)
    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test,"",-1)

    
       # # combined echonest plus date recorded plus simple
    # sample = top_echonest_tracks(True)
    # print("\n\nTesting for all Echonest Features + date recorded")
    # single_echo_plus_date_and_simple = single_features + ['track_duration','track_listens','track_favorites','days_since_first']
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,single_echo_plus_date_and_simple)
    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test,"",-1)

    #    # # combined echonest plus date recorded plus simple plus songname
    # sample = top_echonest_tracks(True)
    # single_echo_plus_date_and_simple = single_features + ['track_duration','track_listens','track_favorites','days_since_first']
    # procesed_track_names = process_track_names(sample, True, False, True)
    # tf_idf_track_names = vectorise(procesed_track_names,"tfidf")
    # X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,tf_idf_track_names,single_echo_plus_date_and_simple)
    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test,"",-1)

