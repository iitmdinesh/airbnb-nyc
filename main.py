import os.path

import numpy as np
from flask import Flask, request, jsonify
from model.indexer import create_index
from model.search import identify_diverse_results
import pandas as pd

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('corpus/listings.csv.gz', compression='gzip')
if os.path.exists('corpus/meta.npz'):
    npz_data = np.load('corpus/meta.npz')
    ids, embeds = npz_data['ids'], npz_data['embeds']
else:
    ids, embeds = create_index(data)
    np.savez('corpus/meta.npz', ids=ids, embeds=embeds)

# location in meters
location = np.stack((84000 * data['longitude'].values, 111000 * data['latitude'].values), axis=1)


def get_similar_listings(listing_id, num_similar=5, expansion_factor=5.0):
    idx = np.where(ids == listing_id)[0]
    similar_indices = identify_diverse_results(
        embeds,
        np.arange(0, embeds.shape[0]),
        location,
        idx,
        num_similar,
        expansion_factor=expansion_factor,
    )
    similar_listings = data.iloc[similar_indices]
    return similar_listings.to_dict(orient='records')


@app.route('/get_similar_listings', methods=['GET'])
def recommend_similar_listings():
    listing_id = request.args.get('listing_id')
    num_similar = request.args.get('num_similar', 5)
    expansion_factor = request.args.get('expansion_factor', 5.0)
    if listing_id:
        listing_id = int(listing_id)
        num_similar = int(num_similar)
        similar_listings = get_similar_listings(
            listing_id,
            num_similar=num_similar,
            expansion_factor=expansion_factor
        )
        return jsonify(similar_listings)
    else:
        return jsonify({'error': 'Listing ID not provided'})


if __name__ == '__main__':
    app.run(debug=False)
