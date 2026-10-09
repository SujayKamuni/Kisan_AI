from pyngrok import ngrok
from app import app
import sys

# Insert your ngrok authtoken here! 
# You can get it from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token("36i518bRZwUa3aEOtD7YZpb82Vw_5Gk4gtWwisF4FE8gpSWhQ")

if __name__ == "__main__":
    port = 5000
    
    try:
        # Open a ngrok tunnel to the localhost server
        public_url = ngrok.connect(port).public_url
        print("="*60)
        print(f"🚀 Your Flask app is running on ngrok at:")
        print(f"👉 {public_url}")
        print("="*60)
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        print("Please make sure you have installed pyngrok ('pip install pyngrok') and entered your authtoken.")
        sys.exit(1)

    # Start the Flask server
    # use_reloader=False is important so ngrok doesn't spawn two tunnels
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
