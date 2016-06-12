## Report for Double Down Strategy
---
###stragegy
- start date: 1/1/2016
- end date: 31/5/2016
- long at dmat_low, dmat_low - $1, ...
- short at dmat_high, dmat_high + $1, ...
- amount of contract for short and long are 1, 1, 2, 4, 8, 16, 32, 64, ...
- take profit at 3% for long and short (base on the average price)
- once exit, reset the corresponding position to 0 and set dmat value of that day as new entry price
- net position limit is 600
- capital is $ 3M
 
###result
- **return**

![](./return_unit1.png)![](./return_unit10.png)
![](./Accumulative_Return.png)

- **position** 

![](./position_unit1.png)![](./position_unit10.png)
![](./net_position.png)

###analysis