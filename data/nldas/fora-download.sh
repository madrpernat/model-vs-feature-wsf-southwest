#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (sywa): " username
    username=${username:-sywa}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197901.002.grb"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197901.002.grb -w %{http_code} | tail  -1`
    if [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w %{http_code} https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197901.002.grb | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197901.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197902.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197903.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197904.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197905.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197906.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197907.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197908.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197909.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197910.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197911.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1979/NLDAS_FORA0125_M.A197912.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198001.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198002.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198003.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198004.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198005.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198006.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198007.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198008.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198009.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198010.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198011.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1980/NLDAS_FORA0125_M.A198012.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198101.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198102.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198103.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198104.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198105.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198106.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198107.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198108.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198109.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198110.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198111.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1981/NLDAS_FORA0125_M.A198112.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198201.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198202.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198203.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198204.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198205.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198206.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198207.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198208.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198209.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198210.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198211.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1982/NLDAS_FORA0125_M.A198212.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198301.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198302.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198303.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198304.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198305.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198306.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198307.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198308.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198309.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198310.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198311.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1983/NLDAS_FORA0125_M.A198312.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198401.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198402.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198403.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198404.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198405.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198406.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198407.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198408.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198409.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198410.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198411.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1984/NLDAS_FORA0125_M.A198412.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198501.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198502.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198503.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198504.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198505.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198506.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198507.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198508.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198509.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198510.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198511.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1985/NLDAS_FORA0125_M.A198512.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198601.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198602.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198603.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198604.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198605.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198606.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198607.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198608.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198609.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198610.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198611.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1986/NLDAS_FORA0125_M.A198612.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198701.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198702.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198703.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198704.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198705.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198706.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198707.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198708.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198709.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198710.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198711.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1987/NLDAS_FORA0125_M.A198712.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198801.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198802.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198803.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198804.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198805.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198806.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198807.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198808.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198809.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198810.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198811.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1988/NLDAS_FORA0125_M.A198812.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198901.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198902.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198903.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198904.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198905.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198906.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198907.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198908.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198909.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198910.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198911.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1989/NLDAS_FORA0125_M.A198912.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199001.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199002.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199003.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199004.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199005.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199006.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199007.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199008.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199009.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199010.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199011.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1990/NLDAS_FORA0125_M.A199012.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199101.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199102.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199103.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199104.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199105.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199106.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199107.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199108.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199109.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199110.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199111.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1991/NLDAS_FORA0125_M.A199112.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199201.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199202.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199203.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199204.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199205.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199206.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199207.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199208.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199209.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199210.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199211.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1992/NLDAS_FORA0125_M.A199212.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199301.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199302.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199303.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199304.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199305.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199306.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199307.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199308.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199309.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199310.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199311.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1993/NLDAS_FORA0125_M.A199312.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199401.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199402.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199403.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199404.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199405.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199406.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199407.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199408.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199409.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199410.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199411.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1994/NLDAS_FORA0125_M.A199412.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199501.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199502.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199503.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199504.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199505.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199506.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199507.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199508.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199509.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199510.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199511.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1995/NLDAS_FORA0125_M.A199512.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199601.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199602.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199603.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199604.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199605.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199606.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199607.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199608.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199609.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199610.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199611.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1996/NLDAS_FORA0125_M.A199612.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199701.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199702.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199703.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199704.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199705.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199706.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199707.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199708.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199709.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199710.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199711.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1997/NLDAS_FORA0125_M.A199712.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199801.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199802.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199803.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199804.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199805.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199806.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199807.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199808.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199809.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199810.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199811.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1998/NLDAS_FORA0125_M.A199812.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199901.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199902.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199903.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199904.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199905.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199906.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199907.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199908.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199909.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199910.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199911.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/1999/NLDAS_FORA0125_M.A199912.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200001.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200002.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200003.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200004.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200005.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200006.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200007.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200008.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200009.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200010.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200011.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2000/NLDAS_FORA0125_M.A200012.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200101.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200102.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200103.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200104.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200105.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200106.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200107.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200108.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200109.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200110.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200111.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2001/NLDAS_FORA0125_M.A200112.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200201.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200202.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200203.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200204.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200205.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200206.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200207.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200208.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200209.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200210.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200211.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2002/NLDAS_FORA0125_M.A200212.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200301.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200302.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200303.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200304.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200305.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200306.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200307.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200308.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200309.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200310.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200311.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2003/NLDAS_FORA0125_M.A200312.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200401.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200402.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200403.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200404.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200405.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200406.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200407.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200408.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200409.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200410.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200411.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2004/NLDAS_FORA0125_M.A200412.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200501.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200502.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200503.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200504.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200505.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200506.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200507.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200508.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200509.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200510.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200511.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2005/NLDAS_FORA0125_M.A200512.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200601.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200602.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200603.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200604.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200605.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200606.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200607.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200608.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200609.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200610.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200611.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2006/NLDAS_FORA0125_M.A200612.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200701.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200702.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200703.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200704.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200705.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200706.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200707.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200708.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200709.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200710.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200711.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2007/NLDAS_FORA0125_M.A200712.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200801.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200802.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200803.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200804.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200805.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200806.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200807.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200808.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200809.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200810.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200811.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2008/NLDAS_FORA0125_M.A200812.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200901.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200902.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200903.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200904.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200905.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200906.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200907.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200908.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200909.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200910.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200911.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2009/NLDAS_FORA0125_M.A200912.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201001.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201002.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201003.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201004.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201005.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201006.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201007.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201008.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201009.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201010.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201011.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2010/NLDAS_FORA0125_M.A201012.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201101.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201102.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201103.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201104.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201105.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201106.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201107.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201108.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201109.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201110.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201111.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2011/NLDAS_FORA0125_M.A201112.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201201.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201202.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201203.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201204.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201205.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201206.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201207.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201208.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201209.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201210.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201211.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2012/NLDAS_FORA0125_M.A201212.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201301.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201302.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201303.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201304.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201305.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201306.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201307.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201308.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201309.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201310.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201311.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2013/NLDAS_FORA0125_M.A201312.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201401.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201402.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201403.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201404.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201405.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201406.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201407.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201408.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201409.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201410.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201411.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2014/NLDAS_FORA0125_M.A201412.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201501.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201502.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201503.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201504.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201505.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201506.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201507.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201508.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201509.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201510.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201511.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2015/NLDAS_FORA0125_M.A201512.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201601.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201602.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201603.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201604.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201605.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201606.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201607.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201608.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201609.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201610.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201611.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2016/NLDAS_FORA0125_M.A201612.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201701.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201702.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201703.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201704.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201705.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201706.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201707.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201708.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201709.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201710.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201711.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2017/NLDAS_FORA0125_M.A201712.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201801.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201802.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201803.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201804.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201805.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201806.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201807.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201808.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201809.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201810.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201811.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2018/NLDAS_FORA0125_M.A201812.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201901.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201902.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201903.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201904.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201905.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201906.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201907.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201908.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201909.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201910.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201911.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2019/NLDAS_FORA0125_M.A201912.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202001.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202002.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202003.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202004.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202005.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202006.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202007.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202008.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202009.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202010.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202011.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2020/NLDAS_FORA0125_M.A202012.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202101.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202102.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202103.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202104.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202105.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202106.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202107.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202108.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202109.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202110.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202111.002.grb
https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_M.002/2021/NLDAS_FORA0125_M.A202112.002.grb
EDSCEOF