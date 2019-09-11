# README

## BloxyGraphCrawler.py

We craw the tx basic information and traces of tx from www.bloxy.info. Here is our source code.

## MethodABICrawler.py

We craw all reversed method bytes and method names from ABI.

## StateofthedappsCrawler.py

We craw https://www.stateofthedapps.com/rankings/platform/ethereum to get famous dapps' detail, including dapps' address and description of dapps. www.dapp.com/ranking

## data

These files are original reports we considered and form our measurement dataset.

The "reports used in measurement.pdf" is original reports and we extracted useful information from them.

We extracted transactions from reports and extened transactions from them to form our dataset. The ExtendTx.csv.zip is our measuement data.

**ExtendTx.csv.zip** is the measurement data.  

> `Group` field represents different incidents and `stage` field indicates the evolutoin of each incidents. (0 means attack preparation, 1 means exploitation, 2 means attack propagation and 3 means mission completion)



| 名称             | 类型       | 描述                                                    |
| ---------------- | ---------- | ------------------------------------------------------- |
| event            | varchar    | report name                                             |
| eventDate        | varchar    | event time period                                       |
| game             | varchar    | game being reported                                     |
| group            | varchar    | the incident name (named by ourselves)                  |
| attackType       | varchar    | attack type                                             |
| seed             | varchar    | seed tx referenced in report                            |
| hashKey          | varchar    | tx hash referenced in report or extended from other txs |
| txDate           | datetime   | tx time                                                 |
| sender           | varchar    | sender smart contract of tx                             |
| senderType       | varchar    | type of sender: smart contract or user                  |
| senderName       | varchar    | Nick name of sender smart contract                      |
| receiver         | varchar    | Receiver smart contract of tx                           |
| receiverType     | varchar    | type of receiver：smart contract or user                |
| receiverName     | varchar    | Nick name of receiver smart contract                    |
| txMethod         | varchar    | method being called in tx                               |
| value            | varchar    | Eth value being tranfered in tx                         |
| input            | mediumtext | input of tx                                             |
| gas              | varchar    | biggest gas set up in tx                                |
| gasPrice         | varchar    | price of biggest gas                                    |
| isError          | varchar    | If tx is error                                          |
| txreceipt_status | varchar    | Tx receipt status                                       |
| contractAddress  | varchar    | created contract address                                |
| gasUsed          | varchar    | gas being used in tx                                    |
| nonce            | varchar    | nonce number in block                                   |
| stage            | varchar    | attack stage                                            |
| graphPosition    | varchar    | tx graph stored position                                |
| traceNum         | int        | trace number in tx                                      |
| gameName         | longtxt    | game name related in traces                             |
| gameAddress      | longtxt    | game address related in traces                          |



**EthereumGame.csv.zip** is the dapp data which contains game name, game address, homepage, classification, and detail discription. It's collected from <https://www.stateofthedapps.com/> at Dec 2018.



**reports used in measurement.pdf**  including reports we used in measuremnt.

**more reports.pdf** contains both reports we used and not used in measurement.

**2018 ethereum attack incidents summary table.xlsx** is a supplement for above two.



