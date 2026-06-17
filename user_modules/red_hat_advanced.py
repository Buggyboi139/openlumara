import core
import asyncio
import subprocess
import re

class RedHatAdvanced(core.module.Module):
    """
    Advanced RedHat Toolkit (RedHat Plus).
    Integrates ProjectDiscovery ecosystem tools (Subfinder, HTTPX, Nuclei).
    Strictly adheres to ROE with built-in Rate Limiting (RL) and concurrency controls.
    """

    settings = {}

    # ---------------------------------------------------------
    # Security Validators
    # ---------------------------------------------------------
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Strict regex for domains/IPs. No paths, no protocols."""
        return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\.-]*$', domain))

    def _is_valid_url(self, url: str) -> bool:
        """Strict regex for URLs."""
        return bool(re.match(r'^https?:\/\/[a-zA-Z0-9\.\-:]+[\/a-zA-Z0-9_\-\.\?\&]*$', url))

    # ---------------------------------------------------------
    # Advanced AI Tools (ProjectDiscovery)
    # ---------------------------------------------------------

    async def pd_subfinder(self, domain: str, profile: str = "fast"):
        """
        AI TOOL: pd_subfinder
        Passive subdomain enumeration using Subfinder. Purely OSINT, touches no target servers.
        
        Args:
            domain: The base domain to scan (e.g., 'example.com').
            profile: Choose from:
                - 'fast': Default passive sources. Fast and lightweight.
                - 'thorough': Queries ALL available passive sources (slower).
                - 'recursive': Attempts to recursively find subdomains of subdomains (slowest).
        """
        if not self._is_valid_domain(domain):
            return "Error: Invalid domain format. Do not include http:// or paths."

        # ROE: Subfinder is passive, but we limit threads (-t) so we don't spam OSINT.
        profiles = {
            "fast": ["-t", "5"],
            "thorough": ["-all", "-t", "5"],
            "recursive": ["-recursive", "-t", "5"]
        }
        
        flags = profiles.get(profile, profiles["fast"])

        def _run():
            try:
                cmd = ["subfinder", "-d", domain, "-silent"] + flags
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                output = result.stdout.strip()
                if not output:
                    return f"No subdomains found for {domain} under profile '{profile}'."

                line_count = len(output.splitlines())
                header = f"--- Subfinder ({profile}) Results for {domain} (Found {line_count}) ---\n"
                full_result = header + output

                if len(full_result) > 1500:
                    return full_result[:1500] + "\n...[TRUNCATED to save context window]"
                return full_result
                
            except FileNotFoundError:
                return "Error: 'subfinder' binary is missing."
            except subprocess.TimeoutExpired:
                return "Error: Subfinder timed out after 60 seconds."
            except Exception as e:
                return f"Subfinder failed: {str(e)}"

        return await asyncio.to_thread(_run)

    async def pd_httpx(self, target: str, profile: str = "polite"):
        """
        AI TOOL: pd_httpx
        Active web probing to detect live servers, titles, and tech stacks.
        
        Args:
            target: The domain or URL to probe.
            profile: Choose from:
                - 'polite': 1 req/sec, minimal probes. Maximum stealth/ROE safety.
                - 'standard': 10 req/sec, pulls tech stack and status codes.
                - 'deep': 5 req/sec, grabs TLS certs and CSP headers.
        """
        if not self._is_valid_domain(target) and not self._is_valid_url(target):
            return "Error: Invalid target format."

        # ROE: Strict Rate Limiting (-rl) and Threads (-t)
        profiles = {
            "polite": ["-rl", "1", "-t", "1", "-title", "-status-code"],
            "standard": ["-rl", "10", "-t", "5", "-title", "-tech-detect", "-status-code"],
            "deep": ["-rl", "5", "-t", "2", "-title", "-tech-detect", "-tls-grab", "-csp-probe"]
        }
        
        flags = profiles.get(profile, profiles["polite"])

        def _run():
            try:
                cmd = ["httpx", "-u", target, "-silent"] + flags
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
                
                output = result.stdout.strip()
                if not output:
                    return f"HTTPX ({profile}) - Target {target} appears dead or unreachable."

                if len(output) > 1500:
                    return output[:1500] + "\n...[TRUNCATED]"
                return output
                
            except FileNotFoundError:
                return "Error: 'httpx' binary is missing."
            except subprocess.TimeoutExpired:
                return "Error: HTTPX timed out after 45 seconds."
            except Exception as e:
                return f"HTTPX failed: {str(e)}"

        return await asyncio.to_thread(_run)

    async def pd_nuclei_scan(self, target: str, profile: str = "safe_recon"):
        """
        AI TOOL: pd_nuclei_scan
        Targeted vulnerability and misconfiguration scanning.
        
        Args:
            target: The URL or domain to scan (e.g., 'https://example.com').
            profile: Choose from:
                - 'safe_recon': Passive header checks and basic tech detection (RL: 5/sec).
                - 'panels': Looks for exposed admin panels and default logins (RL: 5/sec).
                - 'cves': Scans for known vulnerabilities. More noisy but strictly rate limited (RL: 10/sec).
        """
        if not self._is_valid_domain(target) and not self._is_valid_url(target):
            return "Error: Invalid target format."

        # ROE: Strict mapping of tags, Rate Limits (-rl), and Concurrency (-c). 
        # Prevents AI from executing DDoS-like fuzzing templates.
        profiles = {
            "safe_recon": ["-tags", "misconfiguration,tech", "-rl", "5", "-c", "2"],
            "panels": ["-tags", "panel,default-login,exposed-panels", "-rl", "5", "-c", "2"],
            "cves": ["-tags", "cve", "-severity", "low,medium,high,critical", "-rl", "10", "-c", "5"]
        }
        
        flags = profiles.get(profile, profiles["safe_recon"])

        def _run():
            try:
                cmd = ["nuclei", "-u", target, "-silent"] + flags
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                output = result.stdout.strip()
                if not output:
                    return f"Nuclei Scan ({profile}) completed. No findings for {target}."

                header = f"--- Nuclei Findings ({profile}) for {target} ---\n"
                full_result = header + output

                if len(full_result) > 1500:
                    return full_result[:1500] + "\n...[TRUNCATED]"
                return full_result
                
            except FileNotFoundError:
                return "Error: 'nuclei' binary is missing."
            except subprocess.TimeoutExpired:
                return "Error: Nuclei scan timed out. The target may be rate-limiting you."
            except Exception as e:
                return f"Nuclei failed: {str(e)}"

        return await asyncio.to_thread(_run)

    # ---------------------------------------------------------
    # User Commands
    # ---------------------------------------------------------

    @core.module.command("pd_subfinder")
    async def manual_subfinder_cmd(self, args: list):
        """Usage: /pd_subfinder <domain> <fast|thorough|recursive>"""
        if not args: return "Please provide a domain."
        profile = args[1] if len(args) > 1 else "fast"
        return await self.pd_subfinder(args[0], profile)

    @core.module.command("pd_httpx")
    async def manual_httpx_cmd(self, args: list):
        """Usage: /pd_httpx <target> <polite|standard|deep>"""
        if not args: return "Please provide a target."
        profile = args[1] if len(args) > 1 else "polite"
        return await self.pd_httpx(args[0], profile)

    @core.module.command("pd_nuclei")
    async def manual_nuclei_cmd(self, args: list):
        """Usage: /pd_nuclei <target> <safe_recon|panels|cves>"""
        if not args: return "Please provide a target."
        profile = args[1] if len(args) > 1 else "safe_recon"
        return await self.pd_nuclei_scan(args[0], profile)
