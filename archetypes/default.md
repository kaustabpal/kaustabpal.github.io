+++
date = '{{ .Date }}'
draft = true
math = true
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
url = '/{{ .File.ContentBaseName }}'
+++