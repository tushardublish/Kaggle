# Kaggle Airline Price Optimization

https://www.kaggle.com/tdublish3/exercise-airline-price-optimization-microchalleng

Main crux was to understand that expected future demand across all possibilities could be taken as 150.

So, the total revenue equation becomes:

y = (d-x)x + (150 - k/n)*(k/n)*n

where,

x is the seats sold on the current day

k is total_tickets - x

n is days_left-1

For remaining days, we sell the remaining seats equally on every day to maximize profit, as we can correctly assume the expected demand to be 150 everyday
