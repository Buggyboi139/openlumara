import core
import asyncio
import base64
import urllib.parse
import socket
import requests
import subprocess
import re
import ssl
from datetime import datetime

class RedHat(core.module.Module):
    """
    The RedHat Toolkit.
    Provides encoding, decoding, basic network recon, nmap scanning, Shodan, and OSINT.
    """

    # 1. SETTINGS
    settings = {
        "shodan_api_key": {
            "default": "xxxxxx",
            "description": "Your Shodan API key for advanced network recon."
        }
    }
    dependencies = ["requests", "shodan"]

    # 2. INITIALIZATION
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        api_key = self.config.get("shodan_api_key")
        if not api_key:
            self.shodan_api_key = "xxxxxx"
        else:
            self.shodan_api_key = api_key

    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------
    
    def _is_valid_target(self, target: str) -> bool:
        """Strict regex to allow only valid IPs or standard domain names."""
        return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\.-]*$', target))

    # ---------------------------------------------------------
    # Core Utilities
    # ---------------------------------------------------------

    async def encode_decode(self, text: str, action: str, format_type: str):
        """
        AI TOOL: encode_decode
        Encode or decode text payloads. Useful for crafting or analyzing payloads.
        
        Args:
            text: The text to process.
            action: Either 'encode' or 'decode'.
            format_type: The format to use ('base64', 'url', 'hex').
        """
        try:
            if format_type.lower() == 'base64':
                if action == 'encode':
                    return base64.b64encode(text.encode()).decode()
                elif action == 'decode':
                    return base64.b64decode(text.encode()).decode()
            elif format_type.lower() == 'url':
                if action == 'encode':
                    return urllib.parse.quote(text)
                elif action == 'decode':
                    return urllib.parse.unquote(text)
            elif format_type.lower() == 'hex':
                if action == 'encode':
                    return text.encode().hex()
                elif action == 'decode':
                    return bytes.fromhex(text).decode()
            return "Error: Unknown format_type. Use 'base64', 'url', or 'hex'."
        except Exception as e:
            return f"Operation failed: {str(e)}"

    async def dns_lookup(self, domain: str):
        """
        AI TOOL: dns_lookup
        Perform a basic DNS and IP lookup on a target domain.
        
        Args:
            domain: The target domain name (e.g., 'example.com'). Do not include http://
        """
        if not self._is_valid_target(domain):
            return "Error: Invalid domain format."

        def _lookup():
            try:
                ip = socket.gethostbyname(domain)
                geo_resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
                if geo_resp.get("status") == "success":
                    return f"Domain: {domain}\nIP: {ip}\nLocation: {geo_resp['city']}, {geo_resp['country']}\nISP: {geo_resp['isp']}"
                return f"Domain: {domain}\nIP: {ip}"
            except Exception as e:
                return f"DNS Lookup failed for {domain}: {str(e)}"

        return await asyncio.to_thread(_lookup)

    # ---------------------------------------------------------
    # New Recon Tools (Safe & Controlled)
    # ---------------------------------------------------------

    async def whois_lookup(self, domain: str):
        """
        AI TOOL: whois_lookup
        Retrieve WHOIS registration data for a domain.
        
        Args:
            domain: The target domain (e.g., 'example.com').
        """
        if not self._is_valid_target(domain):
            return "Error: Invalid domain format."

        def _whois():
            try:
                # Strictly controlled subprocess. Cannot inject flags because domain is verified.
                result = subprocess.run(["whois", domain], capture_output=True, text=True, timeout=15)
                output = result.stdout
                
                if not output:
                    return f"No WHOIS data found or command failed.\nErrors: {result.stderr}"

                # Strip out excessive blank lines
                output = "\n".join([line for line in output.splitlines() if line.strip() and not line.startswith('%')])

                if len(output) > 1500:
                    return output[:1500] + "\n...[TRUNCATED to save context space]"
                return output
            except FileNotFoundError:
                return "Error: The 'whois' binary is not installed on the system."
            except subprocess.TimeoutExpired:
                return "Error: WHOIS lookup timed out."
            except Exception as e:
                return f"WHOIS lookup failed: {str(e)}"

        return await asyncio.to_thread(_whois)

    async def enumerate_subdomains_crt(self, domain: str):
        """
        AI TOOL: enumerate_subdomains_crt
        Passively find subdomains for a target using Certificate Transparency logs (crt.sh).
        Highly stealthy, touches no target infrastructure.
        
        Args:
            domain: The base domain to search (e.g., 'example.com').
        """
        if not self._is_valid_target(domain):
            return "Error: Invalid domain format."

        def _enum():
            try:
                url = f"https://crt.sh/?q=%25.{domain}&output=json"
                headers = {"User-Agent": "Mozilla/5.0 (RedHat Recon Module)"}
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                
                data = resp.json()
                subdomains = set()
                
                for entry in data:
                    name_value = entry.get("name_value", "")
                    # crt.sh can return multiple domains separated by newlines
                    for sub in name_value.splitlines():
                        clean_sub = sub.strip().lower()
                        if clean_sub.endswith(domain) and '*' not in clean_sub:
                            subdomains.add(clean_sub)
                
                if not subdomains:
                    return f"No subdomains found for {domain} in CT logs."

                result = f"Passively enumerated {len(subdomains)} subdomains for {domain}:\n"
                result += "\n".join(sorted(subdomains))

                if len(result) > 1500:
                    return result[:1500] + "\n...[TRUNCATED to save context space]"
                return result
            except Exception as e:
                return f"Subdomain enumeration failed: {str(e)}"

        return await asyncio.to_thread(_enum)

    async def ssl_cert_check(self, target: str):
        """
        AI TOOL: ssl_cert_check
        Inspects the TLS/SSL certificate of a target to find alternate names, issuers, and expiry.
        
        Args:
            target: The domain or IP address to connect to (port 443).
        """
        if not self._is_valid_target(target):
            return "Error: Invalid target format."

        def _check_cert():
            try:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE  # Accept self-signed for recon purposes
                
                with socket.create_connection((target, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=target) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Fallback for self-signed or invalid certs (requires grabbing binary cert and parsing)
                        if not cert:
                            cert = ssl.DER_cert_to_PEM_cert(ssock.getpeercert(binary_form=True))
                            return f"Certificate is self-signed or untrusted. Raw PEM:\n{cert[:500]}..."

                        issuer = dict(x[0] for x in cert.get('issuer', []))
                        subject = dict(x[0] for x in cert.get('subject', []))
                        alt_names = [x[1] for x in cert.get('subjectAltName', [])]
                        
                        result = f"--- SSL Certificate for {target} ---\n"
                        result += f"Subject: {subject.get('commonName', 'Unknown')}\n"
                        result += f"Issuer: {issuer.get('organizationName', 'Unknown')} ({issuer.get('commonName', 'Unknown')})\n"
                        result += f"Valid From: {cert.get('notBefore')}\n"
                        result += f"Valid Until: {cert.get('notAfter')}\n"
                        
                        if alt_names:
                            result += f"Alternate Names (SANs):\n  - " + "\n  - ".join(alt_names)
                            
                        if len(result) > 1500:
                            return result[:1500] + "\n...[TRUNCATED]"
                        return result
            except Exception as e:
                return f"SSL Check failed for {target} (maybe no HTTPS on port 443?): {str(e)}"

        return await asyncio.to_thread(_check_cert)

    # ---------------------------------------------------------
    # Nmap Tools (Strictly Sandboxed)
    # ---------------------------------------------------------

    async def nmap_scan(self, target: str, scan_type: str = "fast"):
        """
        AI TOOL: nmap_scan
        Perform a strictly controlled port scan against an IP or Domain using Nmap.
        
        Args:
            target: The IP address or domain to scan.
            scan_type: Must be 'fast' (top 100 ports), 'common' (top ports + versions), or 'ping_sweep'.
        """
        if not self._is_valid_target(target):
            return "Error: Invalid target format. Only standard domains and IPs are allowed."

        profiles = {
            "fast": ["-F", "-Pn", "--open"],                     
            "common": ["-sV", "-p", "21,22,23,25,53,80,110,139,143,443,445,3389,8080", "-Pn"], 
            "ping_sweep": ["-sn"]                                
        }

        flags = profiles.get(scan_type, profiles["fast"])

        def _run_nmap():
            try:
                cmd = ["nmap"] + flags + [target]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                output = result.stdout
                if result.returncode != 0:
                    output += f"\nNmap errors:\n{result.stderr}"
                
                output = "\n".join([line for line in output.splitlines() if line.strip()])

                if len(output) > 1500:
                    return output[:1500] + "\n...[TRUNCATED to save context space]"
                return output
            
            except FileNotFoundError:
                return "Error: Nmap is not installed."
            except subprocess.TimeoutExpired:
                return "Error: Nmap scan timed out."
            except Exception as e:
                return f"Nmap tool failed: {str(e)}"

        return await asyncio.to_thread(_run_nmap)

    # ---------------------------------------------------------
    # Shodan Tools
    # ---------------------------------------------------------

    async def shodan_host_lookup(self, ip: str):
        """
        AI TOOL: shodan_host_lookup
        Look up an IP address on Shodan to see open ports, vulnerabilities, and services.
        
        Args:
            ip: The target IP address to look up (e.g. '8.8.8.8').
        """
        if self.shodan_api_key == "xxxxxx" or not self.shodan_api_key:
            return "Error: Shodan API key not configured. Please set it in config.yml."

        def _lookup():
            try:
                import shodan
                api = shodan.Shodan(self.shodan_api_key)
                host = api.host(ip)
                
                result = f"IP: {host.get('ip_str')}\n"
                result += f"Organization: {host.get('org', 'n/a')}\n"
                result += f"OS: {host.get('os', 'n/a')}\n"
                result += f"Open Ports: {host.get('ports', [])}\n\n"
                
                for item in host.get('data', []):
                    result += f"Port: {item.get('port')}\n"
                    banner = "\n".join(item.get('data', '').strip().splitlines()[:3])
                    result += f"Banner:\n{banner}\n"
                    result += "-" * 20 + "\n"
                    
                if len(result) > 1500:
                    return result[:1500] + "\n...[TRUNCATED]"
                    
                return result
            except ImportError:
                return "Error: 'shodan' library is missing. Ask the user to run 'pip install shodan'."
            except Exception as e:
                return f"Shodan Host Lookup failed: {str(e)}"
                
        return await asyncio.to_thread(_lookup)

    async def shodan_search(self, query: str, limit: int = 5):
        """
        AI TOOL: shodan_search
        Search Shodan for exposed devices, servers, and banners using a query.
        
        Args:
            query: The Shodan search query (e.g., 'apache', 'port:22', 'http.title:"hacked by"').
            limit: Max number of results to fetch (default 5 to save context space).
        """
        if self.shodan_api_key == "xxxxxx" or not self.shodan_api_key:
            return "Error: Shodan API key not configured. Please set it in config.yml."

        def _search():
            try:
                import shodan
                api = shodan.Shodan(self.shodan_api_key)
                results = api.search(query)
                
                result_text = f"Query: {query}\nTotal Results on Internet: {results.get('total')}\n\n"
                
                for match in results.get('matches', [])[:limit]:
                    result_text += f"IP: {match.get('ip_str')} | Port: {match.get('port')}\n"
                    result_text += f"Org: {match.get('org', 'n/a')}\n"
                    
                    data_snippet = match.get('data', '').splitlines()[0] if match.get('data') else ''
                    result_text += f"Data: {data_snippet}\n"
                    result_text += "-" * 20 + "\n"
                    
                if len(result_text) > 1500:
                    return result_text[:1500] + "\n...[TRUNCATED]"
                    
                return result_text
            except ImportError:
                return "Error: 'shodan' library is missing."
            except Exception as e:
                return f"Shodan Search failed: {str(e)}"
                
        return await asyncio.to_thread(_search)

    # ---------------------------------------------------------
    # User Commands
    # ---------------------------------------------------------
    
    @core.module.command("nmap")
    async def manual_nmap_cmd(self, args: list):
        """Usage: /nmap <ip_or_domain> <fast|common|ping_sweep>"""
        if not args: return "Please provide a target."
        scan_type = args[1] if len(args) > 1 else "fast"
        return await self.nmap_scan(args[0], scan_type)

    @core.module.command("recon")
    async def manual_recon_cmd(self, args: list):
        """Usage: /recon <domain>"""
        if not args: return "Please provide a domain."
        domain = args[0]
        # Runs basic recon sequentially and returns combined output
        whois_data = await self.whois_lookup(domain)
        subdomains = await self.enumerate_subdomains_crt(domain)
        return f"--- WHOIS ---\n{whois_data}\n\n--- SUBDOMAINS ---\n{subdomains}"
