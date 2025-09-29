# Goal
I would like to do an evaluation on Azure AI Search v.s FAISS against hotpotqa benchmark, to compare the performance and all metrics that matters for a retrival service.

# Requirements
- process hopotqa dataset to make each record a document which can be ingested/indexed by Azure AI Search and Python script that use FAISS
- example file at Benchmarks\hotpot_qa\hotpot_example_format.json
- dev set at Benchmarks\hotpot_qa\hotpot_dev_fullwiki_v1.json
- two set of code should be produced: one for azure ai search which include publish docs to service and retrival; another is for the indexing of docs via faiss and retrival from it
- a module should be created for prepare the document to generate fundation documents that contains `context`
- a module should be created for running golden QA queries devrived from the dev dataset with all the metrics recorded for later analysis
-  