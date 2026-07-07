mkdir -p ~/.streamlit
mkdir -p cache data models

echo "\
[general]
email = \"\"
" > ~/.streamlit/credentials.toml

echo "\
[server]
headless = true
enableCORS = false
port = \$PORT
" > ~/.streamlit/config.toml
