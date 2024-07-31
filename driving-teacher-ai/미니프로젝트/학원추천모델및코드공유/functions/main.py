from firebase_functions import https_fn
from firebase_admin import initialize_app
import ai_recommendation
import json

initialize_app()


@https_fn.on_request(region="asia-northeast1")
def event_2024_summer_academy_recommend(req: https_fn.Request) -> https_fn.Response:
    """
    Handle recommendation requests.
    """
    allowed_preferences = ['price', 'service', 'lecture', 'facility', 'distance']

    try:
        lat = float(req.args.get('lat'))
        lng = float(req.args.get('lng'))
        preference = req.args.get('preference')

        if not lat or not lng:
            raise ValueError("Latitude and longitude are required.")

        if preference:
            preference_list = [p.strip() for p in preference.replace("'","").split(",")]
            print(preference_list)
            invalid_preferences = [
                p for p in preference_list if p not in allowed_preferences]
            if invalid_preferences:
                raise ValueError(f"Invalid preferences: {', '.join(
                    invalid_preferences)}. Allowed preferences are: {', '.join(allowed_preferences)}.")
        else:
            preference_list = []

        result_academy_id = ai_recommendation.recommend((lat, lng), preference_list)

        result = json.dumps({
            "academy_id": result_academy_id
        })

        return https_fn.Response(response=result, status=200, headers={
            "Content-Type": "application/json"
        })

    except ValueError as ve:
        return https_fn.Response(f"Invalid input: {ve}", status=400)
    except Exception as e:
        # Log the exception (could be to a file, a monitoring service, etc.)
        print(f"An error occurred: {e}")
        # Return a generic error message to the client
        error= json.dumps({
            "error": str(e)
        })
        return https_fn.Response(response=error, status=500, headers={
            "Content-Type": "application/json"
        })
