import numpy as np
import random as rd
from numba import vectorize, jit, cuda, float64

@jit(nopython=True)
def reset():
    return np.array([6,6,6,6,6,6,6,6,6,6,6,6,4,4,4,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])

reset()

@jit(nopython=True)
def get_list_action(play_state):
    if play_state[95] == 7:
        return np.array([0])
    if play_state[95] == 1:
        return np.array([1,2])
    if play_state[95] == 2:
        if play_state[15] == 1:
            return np.array([1,2,3])
        else:
            return np.array([0,1])
    if play_state[95] == 3:
        return np.array([4,5,6])
    if play_state[95] == 4:
        list_return = [0]
        for act in range(432):
            target = act//144 +1
            con_lai = act%144
            give = con_lai%12
            take = con_lai//12
            if play_state[give + 19] == 0 or play_state[target * 20 + take + 19] == 0:
                continue
            else:
                list_return.append(act + 28)
        return np.array(list_return)
    if play_state[95] == 5:
        list_return = [0]
        # có từ 1 xu trở lên
        if play_state[34] > 0:
            # thẻ 1,2,2_3
            for card in [11,12,13]:
                if play_state[card-11] > 0 and play_state[card + 89] > 0:
                    list_return.append(card)
        # nếu có từ 2 xu trờ lên
        if play_state[34] > 1:
            for card in [14,15,22]:
                # thẻ 3,4,11_12
                if play_state[card-11] > 0 and play_state[card+89] > 0:
                    list_return.append(card)
        # nếu có từ 3 xu trở lên
        if play_state[34] > 2:
            # thẻ 5, 8, 9_10, 10
            for card in [16,18,20,21]:
                if play_state[card-11] > 0 and play_state[card+89] > 0:
                    list_return.append(card)
        # nếu có từ 4 xu trở lên
        if play_state[34] > 3:
            if play_state[15] != 1:
                list_return.append(7)
        # nếu có từ 5 xu trở lên
        if play_state[34] > 4:
            for card in [17]:
                if play_state[card-11] > 0 and play_state[card+89] > 0:
                    list_return.append(card)
        # nếu có từ 6 xu trở lên
        if play_state[34] > 5:
            for card in [19]:
                if play_state[card-11] > 0 and play_state[card+89] > 0:
                    list_return.append(card)
            if play_state[31] == 0:
                list_return.append(23)
        # nếu có từ 7 xu trở lên
        if play_state[34] > 6:
            if play_state[32] == 0:
                list_return.append(24)
        # nếu có từ 8 xu trở lên
        if play_state[34] > 7:
            if play_state[33] == 0:
                list_return.append(25)
        # nếu có từ 10 xu trở lên
        if play_state[34] > 9:
            if play_state[16] == 0:
                list_return.append(8)
        # nếu có từ 16 xu trở lên
        if play_state[34] > 15:
            if play_state[17] == 0:
                list_return.append(9)
        # nếu có từ 22 xu trở lên
        if play_state[34] > 21:
            if play_state[18] == 0:
                list_return.append(10)
        return np.array(list_return)
            
@jit(nopython=True)
def state_to_player(state):
    turn = state[99]%4
    player_state = [state[0]]
    for phantu in state[1:15]:
        player_state.append(phantu)
    self_state = state[15 + 20 * turn: 35 + 20 * turn]
    next_state1 = state[15 + 20 * ((turn+1) % 4): 35 + 20 * ((turn+1) % 4)]
    next_state2 = state[15 + 20 * ((turn+2) % 4): 35 + 20 * ((turn+2) % 4)]
    next_state3 = state[15 + 20 * ((turn+3) % 4): 35 + 20 * ((turn+3) % 4)]
    con_lai = state[95:]
    for phantu in self_state:
        player_state.append(phantu)
    for phantu in next_state1:
        player_state.append(phantu)
    for phantu in next_state2:
        player_state.append(phantu)
    for phantu in next_state3:
        player_state.append(phantu)
    for phantu in con_lai:
        player_state.append(phantu)
    # print(player_state)
    return np.array(player_state)

def random_player0(play_state,file_temp,file_per):
    a = get_list_action(play_state)
    b = rd.randrange(len(a))
    return a[b],file_temp,file_per

def action_player(state,list_player,file_temp,file_per):
    current_player = state[99]%4
    play_state = state_to_player(state)
    played_move,file_temp[current_player],file_per = list_player[current_player](play_state,file_temp[current_player],file_per)
    return played_move,file_temp,file_per

def system_check_end(state):
    for nguoichoi in range(4):
        if state[15 + nguoichoi*20] * state[16 + nguoichoi*20] * state[17 + nguoichoi*20] * state[18 + nguoichoi*20] == 1:
            return nguoichoi
    return - 1

@jit(nopython=True)
def check_victory(state):
    for nguoichoi in range(4):
        if state[15 + nguoichoi*20] * state[16 + nguoichoi*20] * state[17 + nguoichoi*20] * state[18 + nguoichoi*20] == 1:
            if nguoichoi != 0:
                return 0
            else: 
                return 1
    return -1 

def normal_environment(state,list_player,print_mode,file_temp,file_per):
    while system_check_end(state) == -1:
        current_player = state[99]%4
        if state[95] == 0:
            state[95] = 1
            # check xem chọn 1 hay 2 dice
            if state[15 + current_player*20] == 1:
                choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
                if choice == 1:
                    state[96] = rd.randrange(1,7)
                    state[97] = 0
                if choice == 2:
                    state[96] = rd.randrange(1,7)
                    state[97] = rd.randrange(1,7)
            else:
                state[96] = rd.randrange(1,7)
                state[97] = 0
            if print_mode == 1:
                print("kết quả xúc sắc là",state[96],state[97])
            state[95] = 2
            # check xem có reroll không
            if state[18 + current_player*20] == 1:
                choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
                if choice == 1:
                    state[96] = rd.randrange(1,7)
                    state[97] = 0
                    if print_mode == 1:
                        print("kết quả đổ lại xúc sắc là",state[96],state[97])
                if choice == 2:
                    state[96] = rd.randrange(1,7)
                    state[97] = rd.randrange(1,7)
                    if print_mode == 1:
                        print("kết quả đổ lại xúc sắc là",state[96],state[97])
            state[95] = 3
            # giải quyết kết quả xúc xắc
            sum_dice = state[96] + state[97]
            if print_mode == 1:
                print("đổ ra",sum_dice)
            if sum_dice == 1:
                for nguoichoi in range(4):
                    if state[19 + nguoichoi*20] >0:
                        state[34 + nguoichoi*20] += state[19 + nguoichoi*20]
                        if print_mode == 1:
                            print(nguoichoi,"có thêm",state[19 + nguoichoi*20],"xu, tổng là", state[34 + nguoichoi*20])
            if sum_dice == 2:
                for nguoichoi in range(4):
                    if state[20 + nguoichoi*20] >0:
                        state[34 + nguoichoi*20] += state[20 + nguoichoi*20]
                        if print_mode == 1:
                            print(nguoichoi,"có thêm",state[20 + nguoichoi*20],"xu, tổng là", state[34 + nguoichoi*20])
                if state[21 + current_player*20] > 0:
                    state[34 + current_player*20] += state[21 + current_player*20] * (1 + state[16 + current_player*20])
                    if print_mode == 1:
                        print(current_player,"có thêm",state[21 + current_player*20] * (1 + state[16 + current_player*20]),"xu, tổng là", state[34 + current_player*20])
            if sum_dice == 3:
                for next in range(1,4):
                    oppo = (current_player - next)%4
                    if print_mode == 1:
                        print(state[22 + oppo*20])
                    if state[22 + oppo*20] > 0:
                        cost = (1 + state[36 + oppo*20]) * state[22 + oppo*20]
                        real = min(cost, state[34 + current_player*20])
                        state[34 + current_player*20] -= real
                        state[34 + oppo*20] += real
                        if print_mode == 1:
                            print(oppo,"cướp của",current_player,real,"xu")
                if state[21 + current_player*20] > 0:
                    state[34 + current_player*20] += state[21 + current_player*20] * (1 + state[16 + current_player*20])
                    if print_mode == 1:
                        print(current_player,"có thêm",state[21 + current_player*20] * (1 + state[16 + current_player*20]),"xu, tổng là", state[34 + current_player*20])
            if sum_dice == 4:
                if state[23 + current_player*20] > 0:
                    state[34 + current_player*20] += state[23 + current_player*20] * (3 + state[16 + current_player*20])
                    if print_mode == 1:
                        print(current_player,"có thêm",state[23 + current_player*20] * (3 + state[16 + current_player*20]),"xu, tổng là", state[34 + current_player*20])
            if sum_dice == 5:
                for nguoichoi in range(4):
                    if state[24 + nguoichoi*20] >0:
                        state[34 + nguoichoi*20] += state[24 + nguoichoi*20]
                        if print_mode == 1:
                            print(nguoichoi,"có thêm",state[24 + nguoichoi*20],"xu, tổng là", state[34 + nguoichoi*20])
            if sum_dice == 6:
                if state[31 + current_player*20] == 1:
                    for next in range(1,4):
                        oppo = (current_player-next)%4
                        real = min(2,state[34 + oppo*20])
                        state[34 + current_player*20] += real
                        state[34 + oppo*20] -= real
                        if print_mode == 1:
                            print(current_player,"cướp",real,"từ người chơi",oppo)
                if state[32 + current_player*20] == 1:
                    choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
                    oppo = (current_player + 3 - choice)%4
                    real = min(5,state[34 + oppo*20])
                    state[34 + current_player*20] += real
                    state[34 + oppo*20] -= real
                    if print_mode == 1:
                        print(current_player,"cướp",real,"xu từ người chơi",oppo)
                if state[33 + current_player*20] == 1:
                    state[95] = 4
                    choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
                    choice -= 28
                    if choice > 0:
                        target = choice//144 +1
                        oppo = (current_player + target)%4
                        con_lai = choice%144
                        give = con_lai%12
                        take = con_lai//12
                        if print_mode == 1:
                            print(state[give + 19 + current_player*20],state[oppo * 20 + give + 19])
                        state[give + 19 + current_player*20] -= 1
                        state[oppo * 20 + give + 19] += 1
                        state[oppo * 20 + take + 19] -= 1
                        state[take + 19 + current_player*20] += 1
                        if print_mode == 1:
                            print(current_player,"đổi thẻ",give,"lấy thẻ",take,"từ người chơi",oppo)
                            print(state[give + 19 + current_player*20],state[oppo * 20 + give + 19])
            if sum_dice == 7:
                if state[25 + current_player*20] > 0:
                    state[34 + current_player*20] += state[25 + current_player*20] * state[20 + current_player*20]
                    if print_mode == 1:
                        print(current_player,"có thêm",state[25 + current_player*20] * state[20 + current_player*20],"xu, tổng là", state[34 + current_player*20])
            if sum_dice == 8:
                if state[26 + current_player*20] > 0:
                    state[34 + current_player*20] += state[26 + current_player*20] * 3 * (state[24 + current_player*20] + state[27 + current_player*20])
                    if print_mode == 1:
                        print(current_player,"có thêm",state[26 + current_player*20] * 3 * (state[24 + current_player*20] + state[27 + current_player*20]),"xu, tổng là", state[34 + current_player*20])
            if sum_dice == 9:
                for next in range(1,4):
                    oppo = (current_player - next)%4
                    if print_mode == 1:
                        print(state[28 + oppo*20])
                    if state[28 + oppo*20] > 0:
                        cost = (2 + state[16 + oppo*20]) * state[28 + oppo*20]
                        real = min(cost, state[34 + current_player*20])
                        state[34 + current_player*20] -= real
                        state[34 + oppo*20] += real
                        if print_mode == 1:
                            print(oppo,"cướp của",current_player,real,"xu")
                for nguoichoi in range(4):
                    if state[27 + nguoichoi*20] >0:
                        state[34 + nguoichoi*20] += state[27 + nguoichoi*20] * 5
                        if print_mode == 1:
                            print(nguoichoi,"có thêm",state[27 + nguoichoi*20] * 5,"xu, tổng là", state[34 + nguoichoi*20])
            if sum_dice == 10:
                for next in range(1,4):
                    oppo = (current_player - next)%4
                    if print_mode == 1:
                        print(state[28 + oppo*20])
                    if state[28 + oppo*20] > 0:
                        cost = (2 + state[16 + oppo*20]) * state[28 + oppo*20]
                        real = min(cost, state[34 + current_player*20])
                        state[34 + current_player*20] -= real
                        state[34 + oppo*20] += real
                        if print_mode == 1:
                            print(oppo,"cướp của",current_player,real,"xu")
                for nguoichoi in range(4):
                    if state[29 + nguoichoi*20] >0:
                        state[34 + nguoichoi*20] += state[29 + nguoichoi*20] * 3
                        if print_mode == 1:
                            print(nguoichoi,"có thêm",state[29 + nguoichoi*20] * 3,"xu, tổng là", state[34 + nguoichoi*20])
            if sum_dice > 10:
                if state[30 + current_player*20] > 0:
                    state[34 + current_player*20] += state[30 + current_player*20] * 3 * (state[19 + current_player*20] + state[29 + current_player*20])
                    if print_mode == 1:
                        print(current_player,"có thêm",state[30 + current_player*20] * 3 * (state[19 + current_player*20] + state[29 + current_player*20]),"xu, tổng là", state[34 + current_player*20])
            state[95] = 5
        # bắt đầu mua sắm thôi, mệt vl
        if print_mode == 1:
            print(current_player,"thực có",state[34 + current_player*20],"xu")
        new_xu = state_to_player(state)[34]
        if print_mode == 1:
            print(current_player,"nghĩ là có",new_xu,"xu")
        choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
        # mua thẻ 1
        if choice == 11:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -=1
            if print_mode == 1:
                print(current_player,"bỏ 1 xu mua thẻ 1")
                print("số thẻ 1 còn lại là",state[choice - 11])
        # mua thẻ 2
        if choice == 12:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -=1
            if print_mode == 1:
                print(current_player,"bỏ 1 xu mua thẻ 2")
                print("số thẻ 2 còn lại là",state[choice - 11])
        # mua thẻ 2_3
        if choice == 13:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -=1
            if print_mode == 1:
                print(current_player,"bỏ 1 xu mua thẻ _2_3")
                print("số thẻ _2_3 còn lại là",state[choice-11])
        # mua thẻ 3
        if choice == 14:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 2
            if print_mode == 1:
                print(current_player,"bỏ 2 xu mua thẻ _3")
                print("số thẻ _3 còn lại là",state[choice-11])
        # mua thẻ 4
        if choice == 15:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 2
            if print_mode == 1:
                print(current_player,"bỏ 2 xu mua thẻ _4")
                print("số thẻ _4 còn lại là",state[choice-11])
        if choice == 22:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 2
            if print == 1:
                print(current_player,"bỏ 2 xu mua thẻ _11_12")
                print("số thẻ _11_12 còn lại là",state[choice-11])
        # mua thẻ 5
        if choice == 16:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 3
            if print_mode == 1:
                print(current_player,"bỏ 3 xu mua thẻ _5")
                print("số thẻ _5 còn lại là",state[choice-11])
        # mua thẻ 8
        if choice == 18:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 3
            if print_mode == 1:
                print(current_player,"bỏ 3 xu mua thẻ _8")
                print("số thẻ _8 còn lại là",state[choice-11])
        # mua thẻ 9_10
        if choice == 20:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 3
            if print_mode == 1:
                print(current_player,"bỏ 3 xu mua thẻ _9_10")
                print("số thẻ _9_10 còn lại là",state[choice - 11])
        # mua thẻ 10
        if choice == 21:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 3
            if print_mode == 1:
                print(current_player,"bỏ 3 xu mua thẻ _10")
                print("số thẻ _10 còn lại là",state[choice - 11])
        # mua thẻ w1
        if choice == 7:
            state[15 + current_player*20] = 1
            state[34 + current_player*20] -= 4
            if print_mode == 1:
                print(current_player,"bỏ 4 xu lật thẻ w1")
        # mua thẻ 7
        if choice == 17:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 5
            if print_mode == 1:
                print(current_player,"bỏ 5 xu mua thẻ _7")
                print("số thẻ _7 còn lại là",state[choice -11])
        # mua thẻ 9
        if choice == 19:
            state[choice + 8 + current_player*20] += 1
            state[choice - 11] -= 1
            state[choice + 89] = 0
            state[34 + current_player*20] -= 6
            if print_mode == 1:
                print(current_player,"bỏ 6 xu mua thẻ _9")
                print("số thẻ _9 còn lại là",state[choice - 11])    
        # mua thẻ 6_1
        if choice == 23:
            state[31 + current_player*20] = 1
            state[12] -= 1
            state[34 + current_player*20] -= 6
            if print_mode == 1:
                print(current_player,"bỏ 6 xu mua thẻ _6_1")
                print("số thẻ _6_1 còn lại là",state[12])    
        # mua thẻ 6_2
        if choice == 24:
            state[32 + current_player*20] = 1
            state[13] -= 1
            state[34 + current_player*20] -= 7
            if print_mode == 1:
                print(current_player,"bỏ 7 xu mua thẻ _6_2")
                print("số thẻ _6_2 còn lại là",state[13]) 
        # mua thẻ 6_3
        if choice == 25:
            state[33 + current_player*20] = 1
            state[14] -= 1
            state[34 + current_player*20] -= 8
            if print_mode == 1:
                print(current_player,"bỏ 8 xu mua thẻ _6_3")
                print("số thẻ _6_3 còn lại là",state[14])    
        # mua thẻ w2
        if choice == 8:
            state[16 + current_player*20] = 1
            state[34 + current_player*20] -= 10
            if print_mode == 1:
                print(current_player,"bỏ 10 xu lật thẻ w2")
        # mua thẻ w3
        if choice == 9:
            state[17 + current_player*20] = 1
            state[34 + current_player*20] -= 16
            if print_mode == 1:
                print(current_player,"bỏ 16 xu lật thẻ w3")
        # mua thẻ w4
        if choice == 10:
            state[18 + current_player*20] = 1
            state[34 + current_player*20] -= 22
            if print_mode == 1:
                print(current_player,"bỏ 22 xu lật thẻ w4")
        # nếu skip
        if choice == 0:
            state[95] = 0
            if state[17 + current_player*20] == 1 and state[96] == state[97]:
                if print_mode == 1:
                    print(current_player,"được thêm 1 lượt nữa")
                # return state
            else:
                if print_mode == 1:
                    print("hết lượt của",current_player)
                state[99] += 1
                for bought in range(100,112):
                    state[bought] = 1
                # return state
    # end game, cho mỗi người 1 turn dummy nữa
    state[95] = 7
    for nguoichoi in range(4):
        state[99] += 1
        choice,file_temp,file_per = action_player(state,list_player,file_temp,file_per)
    win = system_check_end(state)
    return win,file_temp,file_per

@jit(nopython=True)
def amount_action_space():
    return 460

def normal_main(list_player,times,print_mode):
    count = [0,0,0,0]
    file_per = [0]
    list_randomed = [0,1,2,3]
    for van in range(times):
        rd.shuffle(list_randomed)
        shuffled_players = [list_player[list_randomed[0]],list_player[list_randomed[1]],list_player[list_randomed[2]],list_player[list_randomed[3]]]
        state = reset()
        file_temp = [[],[],[],[]]
        win,file_temp,file_per = normal_environment(state,shuffled_players,print_mode,file_temp,file_per)
        real_winner = list_randomed[win]
        count[real_winner] += 1
    return count,file_per

len(reset())
