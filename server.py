"""Flower server example."""
import flwr as fl

if __name__ == "__main__":

    #strategy = fl.server.strategy.FedAvg(
        #fraction_fit=0.1,  # Sample 10% of available clients for the next round
    #)
    fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=3),
            #strategy=strategy
        )
