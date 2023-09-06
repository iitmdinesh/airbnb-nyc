# airbnb-nyc
Use NYC data from http://insideairbnb.com/get-the-data/

## Getting started

```shell
mkdir corpus && wget http://data.insideairbnb.com/united-states/ny/new-york-city/2023-09-05/data/listings.csv.gz && mv listings.csv.gz corpus/
```

## Start server
The first time this is run will take some time

```shell
python main.py
```

## Fire a request
```shell
curl -X GET "http://localhost:5000/get_similar_listings?listing_id=52702018&num_similar=10" | jq -C . | less -R
```