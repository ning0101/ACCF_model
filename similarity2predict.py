


def predict_caculate(testfile,\
        train_matrix,\
        train_mean_rate_map,\
        train_movie_mean_map,\
        adj_cosine_map_of_neighbors,\
        data_path):
    '''
    process input file
    :param io_file: python tuple(output file, input file)
    :param train_matrix: python 2-d list
    :param train_mean_rate_map: python dictionary
    :param train_movie_mean_map: python dictionary
    :param iuf_train_matrix: python 2-d list
    :return: void
    '''
    test_map = build_test_user_map(testfile)
    num_of_neighbor = 100  ###########why
    # sort the test users
    list_of_test_user_id = sorted(test_map.keys())    

    with open(f"./prediction_result_1m/user_cos_{data_path}.txt", "w") as user_cosine_file, open(f"./prediction_result_1m/user_pearson_{data_path}.txt", "w") as user_pearson_file, open(f"./prediction_result_1m/item_adcos_{data_path}.txt", "w") as item_adc_file:

        for user_id in list_of_test_user_id:
            user = test_map[user_id]
            list_of_unrated_movie = user.get_list_of_allmovie()

            #根據test的userid去計算每個sim 沒有共同movie的sim設為0 所以user_id會有200個對應所有user
            cosine_list_of_neighbors = sorted_find_similar_neighbor_cosine(user_id, train_matrix, test_map)
            #print(len(cosine_list_of_neighbors)) 

            #根據test的userid去計算每個sim 沒有共同movie的sim設為0 所以user_id會有200個對應所有user
            pearson_list_of_neighbors = sorted_find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map)###計算sim單獨一個user
            # print(len(pearson_list_of_neighbors))

            for movie_id in list_of_unrated_movie:
            
                cosine_rating = predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_matrix, test_map, cosine_list_of_neighbors, train_movie_mean_map)

            
                pearson_rating = predict_rating_with_pearson_correlation(user_id, movie_id, train_matrix, test_map, train_mean_rate_map, pearson_list_of_neighbors)

                
                item_based_adj_cosine_rating = predict_rating_with_item_based_adj_cosine(user_id, movie_id, test_map, adj_cosine_map_of_neighbors, train_mean_rate_map)

    
                user_cosine_rating = int(round(cosine_rating))
                out_line = str(user_id) + " " + str(movie_id) + " " + str(user_cosine_rating) + "\n"
                user_cosine_file.write(out_line)

                user_pearson_rating = int(round(pearson_rating))
                out_line = str(user_id) + " " + str(movie_id) + " " + str(user_pearson_rating) + "\n"
                user_pearson_file.write(out_line)

                item_adj_cosine_rating = int(round(item_based_adj_cosine_rating))
                out_line = str(user_id) + " " + str(movie_id) + " " + str(item_adj_cosine_rating) + "\n"
                item_adc_file.write(out_line)