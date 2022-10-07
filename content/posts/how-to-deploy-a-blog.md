---
title: "Deploy a Personal Blog"
date: 2022-10-04T21:41:15-04:00
draft: false
---

## Let's test the effect of the headings!
This is my first blog in hugos and I write it just to test the features in fuji theme.

``` OCaml
(* Test the codes *)
let first_var = "hello hugo";;
```

``` C++
/*
 * This is my self-made template for Codeforces contests!
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <stack>
#include <deque>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <numeric>
using namespace std;

#define int long long
#define endl "\n"

const int modulo = 998244353;

void solve() {

}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0), cout.tie(0);
    int t; cin >> t; int kase = 0;
    while (t--) {
        // cout << "Case #" << ++kase << ": ";
        solve();
    }
    return 0;
}

```

Let's test the math functions:

$ O(log(n)) $
$ E = mc^2 $

You can see that the blog don't support latex yet. I'll fix it later.

## Register and choose my own domain
After consulting a friend I chose to buy a new domain in namesilo. It's a website that looks like a scam website but its service is the best among all the domain providers. You can consult [this article](https://zhuanlan.zhihu.com/p/33921436) for details.

## Why I finally choose Github Pages
In fact I chose Vercel and Cloudflare for blog deployment at first because I think Github Pages doesn't sound very cool. But after struggling with vercel and cloudflare for hours I turned back to Github Pages Because the product of Vercel and Cloudflare are just bare HTML without rendering the theme and I don't know why. Will try to figure that later. I see someone applied Github Action to Vercel to optimize the Vercel workflow in [this post](https://olich.me/post/building-a-personal-blog-with-hugo-and-vercel/). I will try that later and deploy multi replicas on Cloudflare and Vercel.

## What to do
Apart from tidying and personalizing my blog, I'll try the features of Cloudflare of CDN to test the performance of my blog.


