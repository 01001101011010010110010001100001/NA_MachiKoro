import numpy as np
import numpy.random as rd
import random
from numba import vectorize, jit, cuda, float64

# khởi tạo bàn chơi
@jit(nopython=True)
def reset():
    state = np.array([3, 0, 0, 0, 0,0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,4, 0, 0, 0, 0,0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,4, 0, 0, 0, 0,0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 3, 1, 0, 0, 0,0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 3, 1, 0, 0, 0,0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,0,0,0,10,10,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    actions = np.array([a for a in range(1,44)])
    points = np.array([a for a in range(36)])
    rd.shuffle(actions)
    rd.shuffle(points)
    state = np.append(state,actions)
    state = np.append(state,points)
    return state

@jit(nopython=True)
def dict_actions():
    return np.array([[[0, 0, 0, 0],[2, 0, 0, 0]],[[0, 0, 0, 0],[3, 0, 0, 0]],[[0, 0, 0, 0],[4, 0, 0, 0]],[[0, 0, 0, 0],[1, 1, 0, 0]],[[0, 0, 0, 0],[0, 0, 1, 0]],[[0, 0, 0, 0],[2, 1, 0, 0]],[[0, 0, 0, 0],[0, 2, 0, 0]],[[0, 0, 0, 0],[1, 0, 1, 0]],[[0, 0, 0, 0],[0, 0, 0, 1]],[[2, 0, 0, 0],[0, 2, 0, 0]],[[2, 0, 0, 0],[0, 0, 1, 0]],[[3, 0, 0, 0],[0, 0, 0, 1]],[[3, 0, 0, 0],[0, 3, 0, 0]],[[3, 0, 0, 0],[0, 1, 1, 0]],[[4, 0, 0, 0],[0, 0, 2, 0]],[[4, 0, 0, 0],[0, 0, 1, 1]],[[5, 0, 0, 0],[0, 0, 0, 2]],[[5, 0, 0, 0],[0, 0, 3, 0]],[[0, 1, 0, 0],[3, 0, 0, 0]],[[0, 2, 0, 0],[0, 0, 2, 0]],[[0, 2, 0, 0],[3, 0, 1, 0]],[[0, 2, 0, 0],[2, 0, 0, 1]],[[0, 3, 0, 0],[0, 0, 3, 0]],[[0, 3, 0, 0],[0, 0, 0, 2]],[[0, 3, 0, 0],[1, 0, 1, 1]],[[0, 3, 0, 0],[2, 0, 2, 0]],[[0, 0, 1, 0],[4, 1, 0, 0]],[[0, 0, 1, 0],[1, 2, 0, 0]],[[0, 0, 1, 0],[0, 2, 0, 0]],[[0, 0, 2, 0],[2, 1, 0, 1]],[[0, 0, 2, 0],[0, 0, 0, 2]],[[0, 0, 2, 0],[2, 3, 0, 0]],[[0, 0, 2, 0],[0, 2, 0, 1]],[[0, 0, 3, 0],[0, 0, 0, 3]],[[0, 0, 0, 1],[0, 0, 2, 0]],[[0, 0, 0, 1],[3, 0, 1, 0]],[[0, 0, 0, 1],[0, 3, 0, 0]],[[0, 0, 0, 1],[2, 2, 0, 0]],[[0, 0, 0, 1],[1, 1, 1, 0]],[[0, 0, 0, 2],[1, 1, 3, 0]],[[0, 0, 0, 2],[0, 3, 2, 0]],[[1, 1, 0, 0],[0, 0, 0, 1]],[[2, 0, 1, 0],[0, 0, 0, 2]],[[0, 0, 0, 0],[0, 0, 0, 0]],[[0, 0, 0, 0],[0, 0, 0, 0]]])

@jit(nopython=True)
def dict_points():
    return  np.array([[0, 0, 0, 5],[0,0,2,3],[0,0,3,2],[0,0,0,4],[0,2,0,3],[0,0,5,0],[0,0,2,2],[0,3,0,2],[2,0,0,3],[0,2,3,0],[0,0,4,0],[0,2,0,2],[0,3,2,0],[2,2,0,0],[3,2,0,0],[2,3,0,0],[2,0,2,0],[0,4,0,0],[3,0,2,0],[2,0,0,2],[0,5,0,0],[0,2,2,0],[2,0,3,0],[3,0,0,2],[1,1,1,3],[0,2,2,2],[1,1,3,1],[2,0,2,2],[1,3,1,1],[2,2,0,2],[3,1,1,1],[2,2,2,0],[0,2,1,1],[1,0,2,1],[1,1,1,1],[2,1,0,1]]),np.array([20,18,17,16,16,15,14,14,14,13,12,12,12,6,7,8,8,8,9,10,10,10,11,11,20,19,18,17,16,15,14,13,12,12,12,9])

@jit(nopython=True)
def env_to_player(env_state):
    current_player = env_state[255]%5
    viewpoint = env_state[51*current_player:51*(current_player+1)]
    to_player = env_state[:51*current_player]
    from_player = env_state[51*(current_player+1):255]
    viewpoint = np.append(viewpoint,to_player)
    viewpoint = np.append(viewpoint,from_player)
    bonus = env_state[255:284]
    viewpoint = np.append(viewpoint,bonus)
    deck_a = [-1,-1,-1,-1,-1,-1]
    for vitri in range(6):
        for the in env_state[284:327]:
            if the > -1 and the not in deck_a:
                deck_a[vitri] = the
                break
    deck_p = [-1,-1,-1,-1,-1]
    for vitri in range(5):
        for the in env_state[327:]:
            if the > -1 and the not in deck_p:
                deck_p[vitri] = the
                break
    viewpoint = np.append(viewpoint,deck_a)
    viewpoint = np.append(viewpoint,deck_p)
    return viewpoint

@jit(nopython=True)
def get_list_action(state):
    #phase = 10 là end game
    if state[256] == 10:
        return np.array([65])
    #phase = 0 là trạng thái bình thường
    if state[256] == 0:
        # tạo action 64 mặc định là nghỉ
        list_action = [64]
        # check thẻ action còn dùng được
        data = dict_actions()
        for act in range(6,51):
            if state[act] == -1:
                # check xem chi phí thẻ
                give = data[act-6][0]
                if min(state[:4] - give) >=0:
                    list_action.append(act)
        # thêm thẻ action có thể mua được, action 0 là mua thẻ free, action 1 là mua thẻ mất 1, 5 là thẻ cuối cùng, trả 5 nl
        for act in range(6):
            if act <= np.sum(state[:4]):
                list_action.append(act)
        # thêm thẻ point có thể mua được, 51 là mua thẻ có vàng, 52 là mua thẻ bạc, 53,54,55 là mua thẻ còn lại
        data = dict_points()
        for idp in range(5):
            point_index = state[290+idp]
            cost = data[0][point_index]
            if np.min(state[:4] - cost) >= 0:
                list_action.append(51+idp)
        return np.array(list_action)
    # phase = 1 là trạng thái nâng cấp, 56 là không nâng, 57 nâng vàng, 58 nâng đỏ, 59 nâng xanh
    if state[256] == 1:
        list_action = [65]
        for nl in range(3):
            if state[nl] > 0:
                list_action.append(57+nl)
        return np.array(list_action)
    # phase = 3 là trạng thái đang dùng thẻ, 65 là không dùng nữa
    if state[256] == 3:
        list_action = [65,state[257]]
        return np.array(list_action)
    # phase = 4 là trả nguyên liệu sau khi dùng thẻ, 60 là trả vàng, 63 là trả nâu
    if state[256] > 3:
        list_action = [60,61,62,63]
        for nl in range(4):
            if state[nl] == 0:
                list_action.remove(60+nl)
        return np.array(list_action)
    # phase = 5 là trả nguyên liệu khi mua thẻ action (trả xong được thêm nl bonus trên bàn)

@jit(nopython=True)
def check_win(env_state):
    win = 0
    max = 0
    end = -1
    for nguoichoi in range(5):
        if env_state[nguoichoi*51 + 4] == 5:
            end = 1
        if env_state[nguoichoi*51 +5] > max:
            max = env_state[nguoichoi*51 +5]
            win = nguoichoi
    if end > 0:
        return win
    else:
        return -1

@jit(nopython=True)
def environment(env_state,choice):
    # tìm kiếm người chơi hiện tại
    current_player = env_state[255]%5
    # nếu người chơi nghỉ
    if choice == 64:
        for idc in range(6,51):
            if env_state[current_player*51 + idc] == 1:
                env_state[current_player*51 + idc] = -1
        # print(current_player,"nghỉ")
        env_state[255] += 1
        
        return env_state
    # nếu người chơi chọn mua thẻ action
    if choice in range(6):
        # thêm thẻ vào tay người chơi
        for ida in range(284,327):
            if env_state[ida] > -1:
                card_id = env_state[ida]
                env_state[current_player*51 + card_id + 6] = -1
                # xóa thẻ trên bàn
                env_state[ida] = -1
                #thôi không tìm nữa
                break
        # nếu lấy thẻ free
        if choice == 0:
            env_state[current_player*51:current_player*51+4] += env_state[260:264]
            # nếu lất xong dư nguyên liệu
            if np.sum(env_state[current_player*51:current_player*51+4]) > 10:
                env_state[256] = 4
                env_state[257] = np.sum(env_state[current_player*51:current_player*51+4]) - 10
                # print(current_player,"lấy thẻ và dư nguyên liệu")
                return env_state
            else:
                #nếu không dư nguyên liệu
                # print(current_player,"lấy thẻ nhưng không thừa nl")
                env_state[255] += 1
                
                return env_state
        # nếu lấy thẻ không free
        else:
            #chuyển phase thành 5
            env_state[256] = 5
            # cho số nl cần bỏ vào solving
            env_state[257] = choice
            # xử lý dữ liệu bonus
            # cắt riêng ra
            from_choice = env_state[260+4*choice:260+4*(choice+1)]
            data_bonus = env_state[260+4*(choice+1):284]
            new_data = np.append(data_bonus,from_choice)
            env_state[260+4*choice:284] = new_data
            # print(current_player,"lấy thẻ, đang trả nl")
            return env_state

    # nếu người chơi đánh thẻ action
    if choice in range(6,51):
        # cho thẻ về trạng thái đã dùng(1)
        env_state[choice] = 1
        data = dict_actions()[choice-6]
        give = data[0]
        take = data[1]
        # nếu là thẻ nâng cấp
        if np.sum(take) == 0:
            env_state[256] = 1
            env_state[257] = 52-choice
            # print(current_player,"nâng cấp",52-choice,"lần")
            return env_state
        # nếu là thẻ dùng 1 lần
        if np.sum(give) == 0:
            env_state[current_player*51:current_player*51+4] += take
            # check xem có dư nguyên liệu không
            if np.sum(env_state[current_player*51:current_player*51+4]) > 10:
                # chuyển phase thành 4
                env_state[256] = 4
                env_state[257] = np.sum(env_state[current_player*51:current_player*51+4]) - 10
                # print(current_player,"dùng thẻ nl free, đang thừa nl")
                return env_state
            else:
                # print(current_player,"dùng thẻ free, không thừa nl")
                env_state[255] += 1
                
                return env_state
        # nếu là thẻ dùng nhiều lần
        else:
            env_state[current_player*51:current_player*51+4] += take
            env_state[current_player*51:current_player*51+4] -= give
            # nếu vẫn còn dùng được thì hỏi xem có dùng tiếp không
            if np.min(env_state[current_player*51:current_player*51+4] - give) >= 0:
                env_state[256] = 3
                env_state[257] = choice
                # print(current_player,"dùng thẻ",choice,"vẫn có thể đổi tiếp")
                return env_state
            # nếu đã dùng hết
            else:
                # và thừa 1 đống nguyên liệu
                if np.sum(env_state[current_player*51:current_player*51+4]) > 10:
                    env_state[256] = 4
                    env_state[257] = np.sum(env_state[current_player*51:current_player*51+4]) -10
                    # print(current_player,"sau khi đổi nl còn thừa 1 đống nl")
                    return env_state
                # nếu không thừa nguyên liệu
                else:
                    env_state[256] = 0
                    # print(current_player,"đổi nl xong, hết lượt")
                    env_state[255] += 1
                    return env_state
    # nếu người chơi chọn không dùng thẻ nữa/không nâng cấp nữa
    if choice == 65 or choice == 56:
        # nếu còn thừa nguyên liệu
        if np.sum(env_state[current_player*51:current_player*51+4]) > 10:
            env_state[256] = 4
            env_state[257] = np.sum(env_state[current_player*51:current_player*51+4]) -10
            # print(current_player,"dùng thẻ xong và còn thừa nl")
            return env_state
        # nếu không còn thừa nl 
        else:
            env_state[256] = 0
            env_state[255] += 1
            # print(current_player,"dùng thẻ xong, hết lượt")
            return env_state
    if choice in range(57,60):
        nl_remove = choice - 57
        # nâng cấp nguyên liệu
        env_state[current_player*51 + nl_remove] -= 1
        env_state[current_player*51 + nl_remove + 1] += 1
        # giảm counter
        env_state[257] -= 1
        # nếu đã hết
        if env_state[257] == 0:
            env_state[256] = 0
            env_state[255] += 1
            # print(current_player,"nâng cấp lần cuối, hết lượt")
            return env_state
        else:
            # print(current_player,"nâng cấp",choice -57)
            return env_state
    # nếu nười chơi mua thẻ điểm
    if choice in range(51,56):
        data = dict_points()
        bonus = 0
        if choice == 51:
            if env_state[258] > 0:
                bonus = 3
                env_state[258] -= 1
            else:
                if env_state[259] > 0:
                    bonus = 1
                    env_state[259] -= 1
        if choice == 52:
            if env_state[259] > 0:
                bonus = 1
                env_state[259] -= 1
        # tìm thẻ được mua
        id_tren_ban = choice - 51
        for iddeck in range(36):
            if env_state[327+iddeck] > -1:
                if id_tren_ban == 0:
                    break
                id_tren_ban -= 1
        # vị trí của thẻ trong danh sách
        point_card = env_state[327+iddeck]
        # xóa thẻ khỏi bàn chơi
        env_state[327+iddeck] = -1
        cost = data[0][point_card]
        score = data[1][point_card]
        # print(current_player,"có",env_state[current_player*51:current_player*51+4],"mua thẻ điểm tốn",cost)
        env_state[current_player*51:current_player*51+4] -= cost
        env_state[current_player*51 + 4] += 1
        env_state[current_player*51 + 5] += score + bonus
        env_state[255] += 1
        # print(current_player,"mua điểm, nhận được",score+bonus,"điểm")
        return env_state
    # nếu người chơi trả nguyên liệu
    if choice in range(60,64):
        # trừ nguyên liệu của người chơi
        env_state[current_player*51 + choice-60] -= 1
        env_state[257] -= 1
        # chưa trả hết thì trả tiếp nhé
        if env_state[257] > 0:
            # print(current_player,"trả nguyên liệu",choice-60,"còn",env_state[257],"nữa")
            return env_state
        # nếu trả hết thì check xem đang trả cho cái gì
        else:
            # nếu trả cho việc dùng thẻ
            if env_state[256] == 4:
                env_state[256] = 0
                env_state[255] += 1
                # print(current_player,"trả xong, hết lượt")
                return env_state
            # nếu trả cho việc mua thẻ
            else:
                # cộng nguyên liệu bonus
                env_state[current_player*51:current_player*51+4] += env_state[280:284]
                # reset nguyên liệu bonus
                env_state[280:284] = 0
                # nếu còn thừa nguyên liệu
                if np.sum(env_state[current_player*51:current_player*51+4]) > 10:
                    env_state[256] = 4
                    env_state[257] = np.sum(env_state[current_player*51:current_player*51+4]) -10
                    # print(current_player,"nhận được nl bonus, vượt 10 nl")
                    return env_state
                # nếu không còn thừa nl 
                else:
                    env_state[256] = 0
                    env_state[255] += 1
                    # print(current_player,"trả xong chi phí mua thẻ action")
                    return env_state

@jit(nopython=True)
def amount_action_space():
    return 66


# @jit(nopython=True)
def one_game(list_player,file_per):
    env_state = reset()
    file_temp = [[],[],[],[],[]]
    turn = 0
    while check_win(env_state) == -1:
        current_player = env_state[255]%5
        state = env_to_player(env_state)
        # print(file_temp,current_player)
        action,file_temp[current_player],file_per = list_player[current_player](state,file_temp[current_player],file_per)
        env_state = environment(env_state,action)
        turn += 1
    for turn_bonus in range(5):
        env_state[255] += 1
        current_player = env_state[255]%5
        state = env_to_player(env_state)
        # print("người chơi",current_player,"check victory",check_victory(state))
        # print("hệ thống check win",check_win(env_state))
        action,file_temp[current_player],file_per = list_player[current_player](state,file_temp[current_player],file_per)
    # print("hết ván")
    return check_win(env_state),file_per

def check_victory(state):
    win = -1
    max = 0
    end = -1
    for nguoichoi in range(5):
        if state[nguoichoi*51 + 4] == 5:
            end = 1
        if state[nguoichoi*51 +5] > max:
            max = state[nguoichoi*51 +5]
            win = nguoichoi
    if end > 0:
        if win == 0:
            return 1
        else:
            return 0
    else:
        return -1

# @jit(nopython=True)
def player_random0(state,file_temp,file_per):
    a = get_list_action(state)
    b = random.randrange(len(a))
    return a[b],file_temp,file_per

def normal_main(list_player,times,print_mode):
    count = [0,0,0,0,0]
    file_per = []
    list_randomed = [0,1,2,3,4]
    for van in range(times):
        rd.shuffle(list_randomed)
        shuffled_players = [list_player[list_randomed[0]],list_player[list_randomed[1]],list_player[list_randomed[2]],list_player[list_randomed[3]],list_player[list_randomed[4]]]
        state = reset()
        win,file_per = one_game(shuffled_players,file_per)
        # print(turn)
        real_winner = list_randomed[win]
        count[real_winner] += 1
    return count,file_per

# %timeit one_game
one_game([player_random0,player_random0,player_random0,player_random0,player_random0],[])
