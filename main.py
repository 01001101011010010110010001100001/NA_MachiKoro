def value_player(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    b = rd.randrange(len(a))
    if check_victory(play_state) == 1:
        for idstate in range(len(play_state)):
            name = str(idstate) + "_" + str(play_state[idstate])
            if len(file_per) == 0:
                file_per = {}
            if name not in file_per:
                file_per[name] = 1
            else:
                file_per[name] += 1
    return a[b],file_temp,file_per

def policy_player(play_state,file_temp,file_per):
    if len(file_per) < 2:
        file_per = np.zeros((2,460))
    if len(file_temp) < 1:
        file_temp = np.zeros(460)
    list_action = get_list_action(play_state)
    index_action = rd.randrange(len(list_action))
    score_max = 0
    action = list_action[index_action]
    if np.sum(file_per[1]) > 100000:
        policy = file_per[1]/file_per[0]
        for act in list_action:
            if policy[act] > score_max and act != 0:
                score_max = policy[act]
                action = act
    # else:
    # print(action,file_temp)
    file_temp[action] += 1
    win = check_victory(play_state)
    if win != -1:
        file_per[0] += file_temp
    if win == 1:
        file_per[1] += file_temp
    return action,file_temp,file_per


list_player = [random_player0,random_player0,random_player0,random_player0]
count, data = normal_main(list_player,100,0)
count
