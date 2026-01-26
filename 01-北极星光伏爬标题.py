from __future__ import annotations
import requests
from bs4 import BeautifulSoup, Tag
import time
import argparse
from pathlib import Path
from typing import Optional, List

DEFAULT_URL = "https://guangfu.bjx.com.cn/xm/{}/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# 设置固定的爬取页面范围
start_page = 63
end_page = 79

DEFAULT_OUTPUT_DIR = Path(r"D:\pv\data\txt")

def fetch_html(url: str, *, timeout: int = 10) -> Optional[str]:
    """Retrieve HTML from the remote article page, returning None for 404."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    except requests.RequestException as err:
        raise RuntimeError(f"网络请求失败: {err}") from err

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"请求失败: {response.status_code} {response.reason}. "
            "如目标站点拒绝请求，请尝试调整请求频率或下载HTML后解析。"
        )

    response.encoding = response.apparent_encoding or response.encoding
    return response.text

def extract_titles_from_page(html: str) -> List[str]:
    """Extract titles from the specified cc-list-content element in the page."""
    soup = BeautifulSoup(html, "html.parser")
    title_list = []

    cc_list_content = soup.select_one(".cc-list-content")
    if cc_list_content:
        items = cc_list_content.find_all("li")
        for item in items:
            a_tag = item.find("a")
            if a_tag and a_tag.has_attr('title'):
                title_list.append(a_tag['title'])
    
    return title_list

def scrape_page(page_number: int, output_dir: Path) -> None:
    """抓取指定页面内容，并保存标题"""
    url = DEFAULT_URL.format(page_number)
    print(f"正在抓取页面: {url}")
    html = fetch_html(url)
    if html is None:
        print(f"[MISS] {url} -> 页面不存在")
        return

    titles = extract_titles_from_page(html)
    if titles:
        # 输出标题到文件
        output_path = output_dir / f"page_{page_number}_titles.csv"
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            for title in titles:
                f.write(f"{title}\n")
        print(f"[OK] {url} -> {output_path}")
    else:
        print(f"[SKIP] {url} -> 未提取到任何标题")

def scrape_all_pages(start_page: int, end_page: int, output_dir: Path) -> None:
    """爬取指定范围的页面"""
    for page_number in range(start_page, end_page + 1):
        scrape_page(page_number, output_dir)
        # 可设置适当延迟，避免频繁请求
        time.sleep(2)

if __name__ == "__main__":
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    scrape_all_pages(start_page, end_page, output_dir)
