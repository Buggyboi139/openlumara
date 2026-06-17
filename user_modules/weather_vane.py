import core
import asyncio
import urllib.parse

class WeatherVane(core.module.Module):
    """
    Checks deep meteorological data, forecasts, and astronomical data.
    """

    settings = {}

    async def check_weather(self, location: str):
        """
        Get the current weather conditions, a 3-day forecast, and astronomical data 
        (sunrise, sunset, moon phase, UV index) for any city in the world.
        
        Args:
            location: The name of the city (e.g., 'London', 'New York', 'Tokyo').
        """
        def _fetch():
            # Import inside to prevent module loader bugs!
            import requests 
            
            try:
                safe_location = urllib.parse.quote(location)
                url = f"https://wttr.in/{safe_location}?format=j1"
                
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    return f"Could not find weather data for '{location}'."
                
                data = resp.json()
                
                # 1. Parse Current Conditions
                current = data['current_condition'][0]
                desc = current['weatherDesc'][0]['value']
                temp_f = current['temp_F']
                feels_f = current['FeelsLikeF']
                humidity = current['humidity']
                wind_speed = current['windspeedKmph']
                wind_dir = current['winddir16Point']
                uv_index = current.get('uvIndex', 'N/A')
                visibility = current.get('visibility', 'N/A')
                pressure = current.get('pressure', 'N/A')

                # 2. Parse Astronomy (from today's data)
                today = data['weather'][0]
                astro = today['astronomy'][0]
                sunrise = astro['sunrise']
                sunset = astro['sunset']
                moon_phase = astro['moon_phase']
                moon_illum = astro['moon_illumination']

                # Build the Report
                report = f" **METEOROLOGICAL REPORT FOR: {location.upper()}**\n\n"
                
                report += " **CURRENT CONDITIONS:**\n"
                report += f"- Condition: {desc}\n"
                report += f"- Temperature: {temp_f}°F (Feels like {feels_f}°F)\n"
                report += f"- Wind: {wind_speed} km/h from the {wind_dir}\n"
                report += f"- Humidity: {humidity}%\n"
                report += f"- UV Index: {uv_index}\n"
                report += f"- Visibility: {visibility} km\n"
                report += f"- Pressure: {pressure} hPa\n\n"

                report += " **ASTRONOMY (TODAY):**\n"
                report += f"- Sunrise: {sunrise} | Sunset: {sunset}\n"
                report += f"- Moon: {moon_phase} ({moon_illum}% illuminated)\n\n"

                report += " **3-DAY FORECAST:**\n"
                
                # 3. Parse Forecast Days
                for day in data['weather']:
                    date = day['date']
                    max_f = day['maxtempF']
                    min_f = day['mintempF']
                    
                    # Find the highest chance of rain for the day by checking hourly slots
                    rain_chances = [int(hour.get('chanceofrain', 0)) for hour in day['hourly']]
                    max_rain_chance = max(rain_chances) if rain_chances else 0
                    
                    # Get the general weather description for noon (slot 4 is usually mid-day)
                    day_desc = day['hourly'][4]['weatherDesc'][0]['value']

                    report += f"- **{date}:** {day_desc} | High: {max_f}°F | Low: {min_f}°F | Rain Chance: {max_rain_chance}%\n"

                return report

            except Exception as e:
                return f"Weather lookup failed: {str(e)}"

        return await asyncio.to_thread(_fetch)