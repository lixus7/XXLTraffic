# XXLTraffic

## Data description:

| Dataset        | Data Points | Sensors | Interval | Duration     | Date Range        |
|----------------|-------------|---------|----------|--------------|-------------------|
| **Full_PEMS03**| 2,419,488   | 1,809   | 5 mins   | 23.00 years  | 03/2001 - 03/2024 |
| **Full_PEMS04**| 2,287,872   | 4,089   | 5 mins   | 21.75 years  | 06/2002 - 03/2024 |
| **Full_PEMS05**| 1,998,720   | 573     | 5 mins   | 19.00 years  | 03/2005 - 03/2024 |
| **Full_PEMS06**| 1,945,728   | 705     | 5 mins   | 18.50 years  | 09/2005 - 03/2024 |
| **Full_PEMS07**| 2,287,872   | 4,888   | 5 mins   | 21.75 years  | 06/2002 - 03/2024 |
| **Full_PEMS08**| 2,419,488   | 2,059   | 5 mins   | 23.00 years  | 03/2001 - 03/2024 |
| **Full_PEMS10**| 1,998,720   | 1,378   | 5 mins   | 19.00 years  | 03/2005 - 03/2024 |
| **Full_PEMS11**| 2,261,376   | 1,440   | 5 mins   | 21.50 years  | 09/2002 - 03/2024 |
| **Full_PEMS12**| 2,331,360   | 2,587   | 5 mins   | 22.16 years  | 01/2002 - 03/2024 |

| Datasets(Gap/Hour/Day) | Time Period       | Nodes |
|------------------------|-------------------|-------|
| **PEMS03_Agg**         | 03/2001 - 03/2024 | 151   |
| **PEMS04_Agg**         | 06/2002 - 03/2024 | 822   |
| **PEMS05_Agg**         | 03/2012 - 03/2024 | 103   |
| **PEMS06_Agg**         | 12/2009 - 03/2024 | 130   |
| **PEMS07_Agg**         | 06/2002 - 03/2024 | 1613  |
| **PEMS08_Agg**         | 03/2001 - 03/2024 | 212   |
| **PEMS10_Agg**         | 06/2007 - 03/2024 | 107   |
| **PEMS11_Agg**         | 09/2002 - 03/2024 | 521   |
| **PEMS12_Agg**         | 01/2002 - 03/2024 | 867   |


 
## Data download

#### The sample data include gap data, hourly data and daily data are included in pems05.zip:

- Gap data: pems05_all_common_flow.csv
- Hourly data: pems05_h.csv
- Daily data: pems05_d.csv

How to preprocess the raw data to make gap data, hourly data and daily data is also provided in ```data``` filepath

## Benchmarking
Our benckmarking is based on Time-Series-Library:
- Time-Series-Library: https://github.com/thuml/Time-Series-Library

## License

The XXLTraffic dataset is licensed under CC BY-NC 4.0 International: https://creativecommons.org/licenses/by-nc/4.0. Our code is available under the MIT License: https://opensource.org/licenses/MIT. Please check the official repositories for the licenses of any specific baseline methods used in our codebase.

