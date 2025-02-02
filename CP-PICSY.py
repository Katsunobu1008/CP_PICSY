#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PageRank アルゴリズムと遷移行列（Google Matrix）の可視化 Web アプリケーション

【アーキテクチャ・ソフトウェア設計の原則】
1. Single Responsibility Principle (SRP):
   - 各関数は単一の責務を持ち、グラフ生成、行列計算、PageRank 計算、HTML生成を個別に担当。
2. Separation of Concerns (関心の分離):
   - ビジネスロジックとプレゼンテーションロジックを明確に分離。
3. Modularity (モジュール性):
   - コードを複数の独立した関数に分割し、保守性と拡張性を向上。
4. Parameterization (パラメータ化):
   - ダンピングファクター、Webページ数、最大反復回数、収束誤差、リンク生成確率などのパラメータを外部指定可能に。
5. Readability and Maintainability (可読性と保守性):
   - 豊富なコメントと明瞭な変数・関数名でコードの理解を容易に。
6. Extensibility (拡張性):
   - 新しい機能やアルゴリズムの追加に柔軟に対応可能な設計。
7. Testability (テスト容易性):
   - 各機能を個別の関数に分離し、単体テストが可能。
8. Appropriate Data Structures (適切なデータ構造の利用):
   - グラフは辞書、行列は NumPy 配列を用いて効率的な処理を実現。
9. Clear Separation between Business Logic and Presentation (ビジネスロジックとプレゼンテーションの明確な分離):
   - Flask のルートで計算結果を生成し、HTML テンプレートで表示。
10. Error Handling and Input Validation (エラーハンドリングと入力検証):
    - クエリパラメータの検証を行い、問題があればデフォルト値にフォールバック。

このアプリケーションは、ランダムに生成されたグラフから遷移行列を作成し、PageRank を反復計算により求め、その過程および結果を Web 上で可視化します。
"""

from flask import Flask, request, render_template_string
import numpy as np
import random

app = Flask(__name__)


def generate_random_graph(num_pages, link_prob=0.3):
    """
    ランダムな有向グラフを生成する関数。
    各ノード間のリンクは、指定された確率 link_prob に基づいて作成される。

    Parameters:
        num_pages (int): グラフ内のページ数（ノード数）
        link_prob (float): 各ページ間にリンクが存在する確率 (0.0〜1.0)

    Returns:
        dict: 各ノードのアウトリンクリストを保持するグラフ
              例: { "Page_1": ["Page_2", "Page_3"], "Page_2": ["Page_3"], ... }
    """
    graph = {}
    for i in range(num_pages):
        node = f"Page_{i+1}"
        graph[node] = []
        for j in range(num_pages):
            if i == j:
                continue  # 自己ループは除外
            if random.random() < link_prob:
                graph[node].append(f"Page_{j+1}")
        # 少なくとも1つのアウトリンクが無い場合は、ランダムに1つ追加
        if not graph[node]:
            possible_links = [
                f"Page_{j+1}" for j in range(num_pages) if j != i]
            graph[node].append(random.choice(possible_links))
    return graph


def compute_transition_matrix(graph):
    """
    グラフから遷移行列（Google Matrix）を生成する関数。

    遷移行列 M の構成:
      - 各列 j に対して、ページ j のアウトリンクが存在するならば、
        M[i][j] = 1 / (アウトリンク数)  （i が j からリンクされている場合）
      - もしページ j がアウトリンクを持たない (sink node) 場合は、
        全ての i に対して 1/N を設定する。

    Parameters:
        graph (dict): ノードをキー、アウトリンクのリストを値とするグラフ

    Returns:
        M (numpy.ndarray): 形状 (N x N) の遷移行列
        nodes (list): ノードのリスト（行列のインデックス順）
    """
    nodes = list(graph.keys())
    N = len(nodes)
    M = np.zeros((N, N))
    # ノード名からインデックスへのマッピングを作成
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # 各ノード（列）ごとに値を設定
    for j, node in enumerate(nodes):
        outlinks = graph[node]
        if outlinks:
            # グラフに存在するアウトリンクのみを有効リンクとする
            valid_links = [dest for dest in outlinks if dest in node_to_index]
            if valid_links:
                weight = 1.0 / len(valid_links)
                for dest in valid_links:
                    i = node_to_index[dest]
                    M[i, j] = weight
            else:
                # すべてのアウトリンクが無効な場合 (あり得ないが念のため)
                M[:, j] = 1.0 / N
        else:
            # sink node の場合は全ページに均等に分配
            M[:, j] = 1.0 / N
    return M, nodes


def compute_pagerank(M, d=0.85, tol=1e-6, max_iter=100):
    """
    遷移行列を用いた PageRank の反復計算を行う関数。

    PageRank の反復更新式:
        PR_new = (1-d)/N + d * (M · PR)

    Parameters:
        M (numpy.ndarray): 遷移行列 (形状: N x N)
        d (float): ダンピングファクター（例: 0.85）
        tol (float): 収束判定の許容誤差
        max_iter (int): 最大反復回数

    Returns:
        pr (numpy.ndarray): 最終的な PageRank ベクトル (形状: N)
        iteration_count (int): 収束までに実行された反復回数
    """
    N = M.shape[0]
    # 初期 PageRank は全ノードで均等分布
    pr = np.full(N, 1.0 / N)

    for iteration in range(max_iter):
        pr_new = (1 - d) / N + d * M.dot(pr)
        # 収束判定: ベクトルの L1 ノルムによる変化量が許容誤差未満なら収束
        if np.abs(pr_new - pr).sum() < tol:
            return pr_new, iteration + 1
        pr = pr_new
    return pr, max_iter


@app.route('/')
def index():
    """
    Flask のメインルート。
    クエリパラメータから各種パラメータ（ダンピングファクター、Webページ数、最大反復回数、収束誤差、リンク生成確率）を取得し、
    ランダムグラフの生成、遷移行列の計算、PageRank の反復計算を実施。
    その結果を HTML で可視化して返す。
    """
    # --- クエリパラメータの取得と検証 ---
    try:
        d = float(request.args.get('d', 0.85))
    except ValueError:
        d = 0.85

    try:
        num_pages = int(request.args.get('num_pages', 5))
    except ValueError:
        num_pages = 5

    try:
        max_iter = int(request.args.get('max_iter', 100))
    except ValueError:
        max_iter = 100

    try:
        tol = float(request.args.get('tol', 1e-6))
    except ValueError:
        tol = 1e-6

    try:
        link_prob = float(request.args.get('link_prob', 0.3))
    except ValueError:
        link_prob = 0.3

    random_jump = 1 - d  # ランダムジャンプ確率

    # --- グラフの生成 ---
    graph = generate_random_graph(num_pages, link_prob)

    # --- 遷移行列の計算 ---
    M, nodes = compute_transition_matrix(graph)

    # --- PageRank の計算 ---
    pagerank_vector, iterations = compute_pagerank(M, d, tol, max_iter)

    # --- HTML 用のパラメータ情報 ---
    params_html = "<h2>パラメータ情報</h2><ul>"
    params_html += f"<li>ダンピングファクター (d): {d}</li>"
    params_html += f"<li>ランダムジャンプ確率 (1-d): {random_jump}</li>"
    params_html += f"<li>Webページ数: {num_pages}</li>"
    params_html += f"<li>最大反復回数: {max_iter}</li>"
    params_html += f"<li>収束判定の許容誤差: {tol}</li>"
    params_html += f"<li>リンク生成確率 (link_prob): {link_prob}</li>"
    params_html += "</ul>"

    # --- グラフ情報の可視化 ---
    graph_html = "<h2>グラフのリンク情報</h2>"
    graph_html += "<table border='1' style='border-collapse: collapse;'>"
    graph_html += "<tr><th>Webページ</th><th>アウトリンク</th></tr>"
    for node, links in graph.items():
        links_str = ", ".join(links)
        graph_html += f"<tr><td>{node}</td><td>{links_str}</td></tr>"
    graph_html += "</table>"

    # --- 遷移行列（Google Matrix）の可視化 ---
    matrix_html = "<h2>遷移行列 (Google Matrix)</h2>"
    matrix_html += "<table border='1' style='border-collapse: collapse;'>"
    # ヘッダー行（列のノード名）
    matrix_html += "<tr><th></th>"
    for node in nodes:
        matrix_html += f"<th>{node}</th>"
    matrix_html += "</tr>"
    # 各行に対して数値を表示
    for i, row in enumerate(M):
        matrix_html += f"<tr><th>{nodes[i]}</th>"
        for value in row:
            matrix_html += f"<td>{value:.4f}</td>"
        matrix_html += "</tr>"
    matrix_html += "</table>"

    # --- PageRank 結果の可視化 ---
    pagerank_html = "<h2>PageRank 結果</h2>"
    pagerank_html += f"<p>反復回数: {iterations}</p>"
    pagerank_html += "<table border='1' style='border-collapse: collapse;'>"
    pagerank_html += "<tr><th>Webページ</th><th>PageRank 値</th></tr>"
    for node, rank in zip(nodes, pagerank_vector):
        pagerank_html += f"<tr><td>{node}</td><td>{rank:.6f}</td></tr>"
    pagerank_html += "</table>"

    # --- 統合された HTML テンプレート ---
    html_template = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>PageRank アルゴリズムと遷移行列の可視化</title>
    </head>
    <body>
        <h1>PageRank アルゴリズムの結果と可視化</h1>
        {{ params_html|safe }}
        {{ graph_html|safe }}
        {{ matrix_html|safe }}
        {{ pagerank_html|safe }}
    </body>
    </html>
    """

    return render_template_string(html_template,
                                  params_html=params_html,
                                  graph_html=graph_html,
                                  matrix_html=matrix_html,
                                  pagerank_html=pagerank_html)


if __name__ == '__main__':
    # デバッグモードで Flask サーバーを起動
    app.run(debug=True)
