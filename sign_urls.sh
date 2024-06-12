#!/bin/bash

export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
export http_proxy=fwdproxy:8080
export https_proxy=fwdproxy:8080
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

i=0
for link in $(cat links.txt); do
  echo $i $link
  aws s3 presign s3://codec-avatars-oss/goliath-4/4TB/$link --expires-in 604800 --region us-west-2 --endpoint-url https://s3.us-west-2.amazonaws.com >> signed.txt
  ((++i))
  # ((++i < 10)) || break
done
