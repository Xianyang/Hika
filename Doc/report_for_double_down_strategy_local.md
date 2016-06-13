## Report for Double Down Strategy
---
###stragegy
- start date: 1/1/2016
- end date: 31/5/2016
- long at **dmat_low**, **dmat_low - $1**, ...
- short at **dmat_high**, **dmat_high + $1**, ...
- basal amount of contract for short and long are 1, 1, 2, 4, 8, 16, 32, 64, ... The actual amount is basal amount times unit (**1** and **10** for this report)
- take profit at 3% for long and short (base on the average price)
- once exit, reset the corresponding position to 0 and set dmat value of that day as new entry price
- net position limit is 600
- capital is $ 3M
 
###result
- **return**

![](./return_unit1.png =265x)![](./return_unit10.png =265x)
![](./Accumulative_Return.png =350x)

- **position** 

![](./position_unit1.png =265x)![](./position_unit10.png =265x)
![](./net_position.png =350x)

###analysis
- Position
	
	From 1-5 to 1-22, the price keep going down, so the program keeps taking long position. The program takes a total 512 position for unit 1 and 600 position for unit10 (hit the limit position)
	
	From 3-10 to 5-25, the program takes some short position. The amount is not large
- For the 18.65% return
	
	The exit price is based on the cash flow, net position and take profit percentage. And the return is based on cash flow, net position, exit price and capital. The formula is as follow
	
	`exit price = abs(CF / net position) * (1 + take profit)`
	
	`return = 1000 * (CF - net position * exit price) / capital`
	
	`= abs(CF) * take profit * 1000 / capital = abs(CF) / 100,000`
	
	From the formula, if the CF is bigger and we actually take profit, the return will be higher. 
	
	On 2016-1-22, the CF is -18,651.0939 and we take profit, so the total return for the past long is `abs(CF) / 100,000 = 18.65%`
	