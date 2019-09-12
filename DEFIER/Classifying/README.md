# src



Classifying the transaction clusters

- Integrating transaction cluters of goodset and unknownset and sliding windows of badset and goodset.

- ```
  cd graph_classification
  python prepossesing.py -beforeClassify=1
  ```

- run the classification

- ```
  run.sh
  ```

- write the classfying result into database and statistic the result

- ```
  python prepossesing.py -beforeClassify=0
  ```

