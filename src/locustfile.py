from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_post(self):
        payload = [
            {
                "Classical_frequency": 1,
                "Country_frequency": 0,
                "EDM_frequency": 2,
                "Folk_frequency": 3,
                "Gospel_frequency": 1,
                "Hip hop_frequency": 2,
                "Jazz_frequency": 0,
                "K pop_frequency": 0,
                "Latin_frequency": 0,
                "Lofi_frequency": 2,
                "Metal_frequency": 2,
                "Pop_frequency": 3,
                "R&B_frequency": 1,
                "Rap_frequency": 3,
                "Rock_frequency": 3,
                "Video game music_frequency": 3,
                "Anxiety": 7,
                "Depression": 10,
                "Insomnia": 5,
                "OCD": 6,
                "Age": 13
            }
        ]

        self.client.post("/predict-all", json=payload)
