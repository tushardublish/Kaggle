# Kaggle Blackjack Microchallenge
# https://www.kaggle.com/tdublish3/exercise-blackjack-microchallenge
# Used Dynamic Programming to calculate winning chances

def get_probability(current_total):
    totalcount = 1
    under21count = 0
    if current_total in total_possibilities and current_total in under21_possibilities:
        return total_possibilities[current_total], under21_possibilities[current_total]
    if current_total > 21:
        return 1,0
    else:
        for i in cards:
            if i == 0:
                totalcount += 1
                under21count += 1
            if i > 0:
                # print(current_total, i)
                itotal, iunder21 = get_probability(current_total + i)
                totalcount += itotal
                under21count += iunder21
        total_possibilities[current_total] = totalcount
        under21_possibilities[current_total] = under21count
        return totalcount, under21count

if __name__=="__main__":
    cards = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    # Initializing end scenarios
    total_possibilities = {21: 1, 22: 1}
    under21_possibilities = {21: 0, 22: 0}
    # Going in reverse and storing results for dynamic prgramming
    # When we calculate possibilities for smaller values, we need to have results for larger values
    for i in range(22, 2, -1):
        get_probability(i)

    print("Total Possibilites: ", total_possibilities)
    print("Under 21: ", under21_possibilities)
    print("Winning Chances:")
    for i in range(1,22):
        if i in total_possibilities:
            print(i, (100.0 * under21_possibilities[i]) / total_possibilities[i], "%")
