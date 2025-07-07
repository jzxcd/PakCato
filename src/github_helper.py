import pandas as pd
import json
import os
from .utils import filter_badges, filter_css
import time
from github import Github, GithubException


class GhMetadata:

    def __init__(self, repo_url=None, owner=None, repo=None, verbose=True):
        token = os.environ.get('GITHUB_API_KEY')
        if not token:
            raise RuntimeError("Please set GITHUB_API_KEY in your environment")
        # disable PyGithub’s built-in retry/backoff
        self.gh = Github(token, retry=0)
        self.verbose = verbose

        if repo_url:
            self.owner, self.repo = self._parse_owner_repo(repo_url)
        elif owner and repo:
            self.owner, self.repo = owner, repo
        else:
            raise ValueError("Must provide either repo_url or both owner and repo")

        self._repo_obj = None

    def _parse_owner_repo(self, url):
        url = url.rstrip('/')
        parts = url.split('/')
        if len(parts) < 2:
            raise ValueError("Invalid GitHub repo URL. Format: https://github.com/owner/repo")
        return parts[-2], parts[-1]

    @property
    def _repo(self):
        if self._repo_obj is None:
            api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            try:
                self._repo_obj = self.gh.get_repo(f"{self.owner}/{self.repo}")
            except GithubException as e:
                code = e.status
                if code == 403:
                    if self.verbose:
                        print(f"{code} Error: {api_url}")
                    return None
                # one-time retry on 404
                if code == 404 and not getattr(self, "_repo_404_retried", False):
                    self._repo_404_retried = True
                    if self.verbose:
                        print(f"{code} Error: {api_url}. Retrying...")
                    time.sleep(10)
                    return self._repo  # retry
                if self.verbose:
                    print(f"{code} Error: {api_url}")
                return None
        return self._repo_obj

    def _execute_api(self, call_fn, api_url, default, retries_404=1, retries_500=3):
        """
        call_fn: no-arg lambda that invokes the PyGithub call
        api_url: for logging
        default: value to return on 403 or exhausted retries on 404/500
        """
        count_404 = count_500 = 0

        while True:
            repo = self._repo
            if repo is None:
                return default

            try:
                return call_fn() or default
            except GithubException as e:
                code = e.status
                # 403 → immediate default
                if code == 403:
                    if self.verbose:
                        print(f"{code} Error: {api_url}")
                    return default
                # retry up to retries_404 times
                if code == 404 and count_404 < retries_404:
                    count_404 += 1
                    if self.verbose:
                        print(f"{code} Error: {api_url}. Retrying ({count_404}/{retries_404})...")
                    time.sleep(10)
                    continue
                if code == 404:
                    if self.verbose:
                        print(f"{code} Error: {api_url}")
                    return default
                # retry up to retries_500 times
                if code == 500 and count_500 < retries_500:
                    count_500 += 1
                    if self.verbose:
                        print(f"{code} Error: {api_url}. Retrying ({count_500}/{retries_500})...")
                    time.sleep(10)
                    continue
                if code == 500:
                    if self.verbose:
                        print(f"{code} Error: {api_url}")
                    return default
                # any other status → exception
                raise RuntimeError(f"Error: {code} – {e.data.get('message', str(e))} – {api_url}")

    def get_github_topic(self):
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/topics"
        return self._execute_api(
            call_fn=lambda: self._repo.get_topics(),
            api_url=api_url,
            default=[]
        )

    def get_readme(self):
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/readme"
        raw = self._execute_api(
            call_fn=lambda: self._repo.get_readme().decoded_content.decode('utf-8'),
            api_url=api_url,
            default=""
        )
        # apply your existing filters
        if raw:
            raw = filter_badges(raw)
            raw = filter_css(raw)
        return raw
    

def get_metadata(testset: list) -> pd.DataFrame:
    testset_df = pd.DataFrame(testset)
    readme_list = []
    topic_list = []
    for i, row in testset_df.iterrows():
        gh = GhMetadata(row['url'])
        readme_list.append(gh.get_readme())
        topic_list.append(gh.get_github_topic())
    parsed_testset_df = testset_df.copy()
    parsed_testset_df['readme'] = readme_list
    parsed_testset_df['topic'] = [json.loads(i).get("names") if i else i for i in topic_list]
    return parsed_testset_df.dropna()
