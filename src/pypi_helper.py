import requests
from .utils import filter_badges, filter_css


class PypiMetadata:

    def __init__(self, package_name):
        self.url = f"https://pypi.org/pypi/{package_name}/json"
        self.raw_metadata = self.fetch_json()
                        
    def fetch_json(self):
        try:
            response = requests.get(self.url)
            # Raise an error if the request was unsuccessful
            response.raise_for_status()
            metadata = response.json()
            metadata['status'] = 'success'
            return metadata
        except requests.exceptions.HTTPError as http_err:
            return {'status': str(http_err)}
        except Exception as err:
            return {'status': str(err)}

    def get_description(self, raw_metadata_info):
        # description to be parse
        raw_description = raw_metadata_info.get('description')
        description = filter_badges(raw_description)
        description = filter_css(description)
        return description

    def get_topic(self, raw_metadata_info):
        topics = []
        for i in raw_metadata_info.get('classifiers'):
            if 'Topic ::' in i:
                topics.append(i.split('Topic :: ')[-1])
        return topics

    def get_output(self):
        output = {}
        output['status'] = self.raw_metadata.get('status')
        raw_metadata_info = self.raw_metadata.get('info')

        if not raw_metadata_info:
            return output
        output['description'] = self.get_description(raw_metadata_info)
        output['topic'] = self.get_topic(raw_metadata_info)
        output['keywords'] = raw_metadata_info.get('keywords')
        output['summary'] = raw_metadata_info.get('summary')
        
        return output
