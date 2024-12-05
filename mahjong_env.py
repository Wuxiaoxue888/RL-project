from typing import Optional
import numpy as np
import gymnasium as gym


class MahjongEnv(gym.Env):
    """Custom gym environment class for Mahjong.
    Create an instance of the environemnet by calling "gym.make("Mahjong-v0")".
    
    Methods:
    get_valid_actions()
    reset()
    step(action)
    """

    def __init__(self):
        self._NUMBER_OF_PLAYERS = 4
        self._UNIQUE_CARDS = 3*9 # suits * values. 3: number of different suits in the deck (Circles, Lines, Symbols). 9: values (1-9) for each suit
        self._DUPLICATES = 4 # There exists 4 copies of each card, e.g. 4 cards of 5-Circle and 4 cards of 8-Line
        self._HAND_SIZE = 14 # Every player starts the game with 13 cards. When it becomes a player's turn they will get one card from the deck -> 14 cards in hand

        self._BASIC_WIN_SCORE = 5
        self._LEVEL_1_WIN_SCORE = 20
        self._LEVEL_2_WIN_SCORE = 30
        self._LEVEL_3_WIN_SCORE = 60
        self._KANG_SCORE = 3

        # Set which player will be the AI. The other three players will take automatic random actions until it is the AI's turn again
        self._AI = 0

        # Flag for if the game is running or not. Sets to true in the reset() function.
        self._game_running = False

        # Stores the score for each player calculated at the end of the game. This array is used in _get_info() as auxiliary information
        self._game_result = np.zeros(self._NUMBER_OF_PLAYERS)

        # Creates the deck, total: 108 cards (Cirles1-9, Lines1-9, Symbols1-9), 4 of each
        # Each unique card is represented as 0-26. The deck consists of four duplicates of each card: 27*4 = 108
        self._deck = [n for n in range(self._UNIQUE_CARDS)]*self._DUPLICATES

        # Sets the turn, there are four players (0-3). AI starts.
        self._player_turn = self._AI

        # Creates the board. Each player has: cards in hand, pongs, kangs, and discarded cards
        self._board = {}
        for player in range(self._NUMBER_OF_PLAYERS):
            self._board[player] = {
                "hand": np.zeros(self._UNIQUE_CARDS, dtype=int),
                "pongs": np.zeros(self._UNIQUE_CARDS, dtype=int),
                "kangs": np.zeros(self._UNIQUE_CARDS, dtype=int),
                "discarded": np.zeros(self._UNIQUE_CARDS, dtype=int),
            }

        """ 
        Observations are dictionaries with the current player's hand, pongs and kangs, and all visible cards on the board.
        They are numpy arrays with 27 elements each, one for each unique card.
        An example of a hand looks like this: [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0]. This hand has one Circle-3 and two Line-8 cards.
        The "visible cards" array behaves exactly like the "hand" array, but the pong and kang arrays are different.
        The values in the pong array are only 0 or 1 (since a pong is 3 duplicates it's impossible to have more than one pong of the same unique card).
        The values in the kang array show which player was responsible for the kang. 0 means no kang, 1 means player 1 was responsible for the kang, 2 means player 2 was responsible for the kang and so on.
        Example of a kang array: [0,1,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        The kang array shows that the player has two kangs, Circle-2 and Circle-5, which player 1 and player 4 were responsible for respectively. A player can be responsible for their own kang (which is worth a lor of points)
        """
        self.observation_space = gym.spaces.Dict(
            {
                "hand": gym.spaces.Box(0, self._DUPLICATES, shape=(self._UNIQUE_CARDS,), dtype=int),
                "pongs": gym.spaces.Box(0, 1, shape=(self._UNIQUE_CARDS,), dtype=int),
                "kangs": gym.spaces.Box(0, self._NUMBER_OF_PLAYERS, shape=(self._UNIQUE_CARDS,), dtype=int),
                "visible cards": gym.spaces.Box(0, self._DUPLICATES, shape=(self._UNIQUE_CARDS,), dtype=int),
            }
        )

        # There are 27 possible actions, corresponding to being able to discard each one of the 27 unique cards in the deck.
        # Since a player has a maximum of 14 cards in their hand it means that not all actions are possible at the same time -> actions need to be masked.
        # To get the valid actions in a given state one can call the get_valid_actions() function.
        self.action_space = gym.spaces.Discrete(self._UNIQUE_CARDS)

        # Add methods to super class
        gym.Env.get_valid_actions = self.get_valid_actions

    def _get_obs(self):
        # Collects all visible cards on the board
        visible_cards = np.zeros(self._UNIQUE_CARDS, dtype=int)
        for player in range(self._NUMBER_OF_PLAYERS):
            visible_cards += self._board[player]["discarded"]
            if player != self._player_turn:
                visible_cards += self._board[player]["pongs"] * 3
                visible_cards += (self._board[player]["kangs"]%2) * 4

        assert len(visible_cards) == self._UNIQUE_CARDS
        assert np.max(visible_cards) <= self._DUPLICATES

        return {
            "hand": self._board[self._player_turn]["hand"],
            "pongs": self._board[self._player_turn]["pongs"],
            "kangs": self._board[self._player_turn]["kangs"],
            "visible cards": visible_cards
        }
    
    def _get_info(self):
        return {
            "": None
        }    
    
    def _is_winning_combination(self, player: int, last_picked_card: int) -> int:
        """Check if a player has a winning combination

        Parameters:
        player: the player to check if they have a winning combination
        last_picked_card: the card the player picked last

        Returns 0 if there is no winning combination, otherwise:
        - 5 for a basic combination
        - 20 if there are no straights
        - 30 if all cards are of the same suit, OR the hand consists of 7 pairs, OR only a pair in hand
        - 60 if there are no straights and all cards are of the same suit, OR five pairs in hand and the last picked card is part of a kang
        """
        
        # Get the hand of the player and make a copy of it to avoid modifying it
        hand = self._board[player]["hand"].copy()

        # Check for two special winning combinations
        pairs = np.nonzero((hand==2))[0]
        if len(pairs) == 7: # 7 pairs in hand
            return self._LEVEL_2_WIN_SCORE
        elif len(pairs) == 5 and self._board[player]["kangs"][last_picked_card]: # 5 pairs in hand and last picked card is part of kang
            return self._LEVEL_3_WIN_SCORE

        # Pad each suit with zeros to make checking for straights easier (and avoid index out of bounds error)
        hand = np.insert(hand, obj=[0,9,18,27], values=0)

        assert len(hand) == self._UNIQUE_CARDS + 4

        def next(card_indices, cards, n):
            """Helper function for the recursive "backtrack" function.
            It handles the next recursive call, by removing checked cards from the hand and then adding them back again
            When there are no cards left in the hand it means it's a winning combination
            """
            cards[card_indices] -= n
            if cards.sum() == 0 or backtrack(np.nonzero(cards)[0][0], cards):
                cards[card_indices] += n
                return True
            cards[card_indices] += n
            return False
        
        def backtrack(card_index, cards):
            """Recursive function that checks if the hand has families (three of a kind or straights)
            It finds valid families and removes them from the hand.

            Parameters:
            card_index - Index of the first card left in the hand. A family for this card is searched for.
                         If no family for the card can be found it means that the hand is not a winning hand.
            cards - The cards left in the hand, after iteratively removing families.
            """
            
            # Check for three of a kind
            if cards[card_index] >= 3:
                if next(card_index, cards, 3):
                    return True
            
            # Check for straights
            if cards[card_index-1] > 0:
                if cards[card_index-2] > 0:
                    if next([card_index-2,card_index-1,card_index], cards, 1):
                        return True
                if cards[card_index+1] > 0:
                    if next([card_index-1,card_index,card_index+1], cards, 1):
                        return True
            if cards[card_index+1] > 0 and cards[card_index+2] > 0:
                if next([card_index,card_index+1,card_index+2], cards, 1):
                    return True
                
            return False

        # The winning combinations needs at least one pair. This code finds all possible pairs and loops through them.
        # Each call in the loop starts a recursive search for families.
        pair_indices = np.where(hand>1)[0]
        for pair_index in pair_indices:
            if next(pair_index, hand, 2):
                
                # Check if all cards (hand, pongs, and kangs) are of the same suit
                all_cards = np.concatenate((np.nonzero(self._board[player]["hand"])[0],
                                            np.nonzero(self._board[player]["pongs"])[0],
                                            np.nonzero(self._board[player]["kangs"])[0]))
                same_suit = (all_cards.max() < 9) or (all_cards.min() > 17) or (all_cards.min() > 8 and all_cards.max() < 18)

                # Check if there are no straights
                no_straigths = len(np.where((hand!=3) & (hand!=0))[0]) == 1 and len(np.where(hand==2)[0]) == 1
                
                if same_suit:
                    if no_straigths:
                        return self._LEVEL_3_WIN_SCORE
                    return self._LEVEL_2_WIN_SCORE
                elif no_straigths:
                    if hand.sum() == 2: # if the hand consists of a single pair
                        return self._LEVEL_2_WIN_SCORE
                    return self._LEVEL_1_WIN_SCORE
                else:
                    return self._BASIC_WIN_SCORE
            
        return 0
    
    def _is_ready(self, player: int) -> int:
        """A player is "ready" if they are one card away from a winning combination.
        This function checks if a player is "ready" or not.
        The function works by looping through potential cards the player needs, giving it to the player and checking if the hand is now a winning combination.
        """

        hand = self._board[player]["hand"]
        cards_in_hand = np.nonzero(hand)[0] # The cards in the player's hand

        checked_cards = [] # cards that have already been checked and we now know are not needed for a winning combination

        # Check if the player needs a pong to kang card if he has 5 pairs in hand. This is a special winning combination
        if len(np.nonzero((hand==2))[0]) == 5 and self._board[player]["pongs"].sum() == 1:
            pong_to_kang_card = np.where(self._board[player]["pongs"]==1)[0][0]
            checked_cards.append(pong_to_kang_card)

            self._board[player]["pongs"][pong_to_kang_card] -= 1
            self._board[player]["kangs"][pong_to_kang_card] += 1
            combination_score = self._is_winning_combination(player, pong_to_kang_card)
            self._board[player]["pongs"][pong_to_kang_card] += 1
            self._board[player]["kangs"][pong_to_kang_card] -= 1
            if combination_score:
                return combination_score

        
        for card in cards_in_hand:
            if card in checked_cards:
                continue
            checked_cards.append(card)

            # Check if the player needs a duplicate of the looped card for a pair or a three of a kind
            hand[card] += 1
            combination_score = self._is_winning_combination(player, card)
            hand[card] -= 1
            if combination_score:
                return combination_score

            # Check if the player needs an adjacent card to the looped card for a straight
            cards_for_straight = [] # will consist of one or both of the adjacent cards to the looped card
            if card%9 > 0:
                cards_for_straight.append(card-1)
            if card%9 < 8:
                cards_for_straight.append(card+1)
            for card_for_straight in cards_for_straight:
                if card_for_straight in checked_cards:
                    continue
                checked_cards.append(card_for_straight)

                hand[card_for_straight] += 1
                combination_score = self._is_winning_combination(player, card_for_straight)
                hand[card_for_straight] -= 1
                if combination_score:
                    return combination_score

        return 0
    
    def _calculate_kang_scores(self) -> list[float]:
        """Helper function that calculates the kang score for each player when the game has ended.
        It is used by calculate_win_scores() and calculate_draw_scores()
        
        Returns:
        The number of points each player won or lost based on kangs.
        """
        # The scores from kangs for each player
        kang_scores = np.zeros(self._NUMBER_OF_PLAYERS)

        # Calculate the kang scores for each player
        for player in range(self._NUMBER_OF_PLAYERS):
            kangs = self._board[player]["kangs"] - 1
            # A multiplier that handles if kangs should increase or decrease a player's score depending on if the player is ready or not 
            ready_multiplier = 1 if self._is_ready(player) else -1
            for kang in kangs:
                if kang == player: # player is responsible for his own kang
                    kang_scores -= self._KANG_SCORE * ready_multiplier
                    kang_scores[player] += self._KANG_SCORE * 4 * ready_multiplier
                elif kang >= 0: # another player is responsible for the kang
                    kang_scores[kang] -= self._KANG_SCORE * ready_multiplier
                    kang_scores[player] += self._KANG_SCORE * ready_multiplier

        return kang_scores
    
    def _calculate_win_scores(self, winning_player: int, combination_score: int, responsible_player: int) -> list[float]:
        """Calculates the score for each player when a player has gotten a winning combination and the game is over.

        Parameters:
        winning_player - the player who got the winning combination
        combination_score - the score of the winning combination
        responsible_player - the player responsible for the winning combination.
                             If a player discarded a card that made the winning player get the winning combination, then that player is responsible.
                             If a player gets a winning combination by picking the last card himself, he is responsible for his own win.
        
        Returns:
        The number of points each player won or lost.
        """
        # The score for each player
        scores = np.zeros(self._NUMBER_OF_PLAYERS)

        # If a player is responsible for his own win, then everyone "pays" him
        if winning_player == responsible_player:
            scores -= combination_score
            scores[winning_player] += combination_score * 4
        else: # only the player responsible for the win "pays" the player with the winning combination
            scores[responsible_player] -= combination_score
            scores[winning_player] += combination_score
        
        # Add the kang scores
        scores += self._calculate_kang_scores()
        
        return scores
    
    def _calculate_draw_scores(self) -> list[float]:
        """Calculates the score for each player when the game has ended in a draw.
        
        Returns:
        The number of points each player won or lost.
        """
        # The score for each player
        scores = np.zeros(self._NUMBER_OF_PLAYERS)

        # List of if players are ready or not
        ready_players = [self._is_ready(player) for player in range(self._NUMBER_OF_PLAYERS)]

        # If all players are ready or no or no one is ready then no one loses or wins any points
        if all(ready_players) or not any(ready_players):
            return scores
        
        # The players who are not ready has to "pay" the players who are ready
        for player in range(self._NUMBER_OF_PLAYERS):
            if ready_players[player]:
                continue
            # "Pay" all players that are ready. The amount of points "payed" is based on the winning combination score.
            # The ready_players array is zero for the players that are not ready. For the players that are ready it shows the combination score they are ready for.
            scores += ready_players
            scores[player] -= np.array(ready_players).sum()
        
        # Add the kang scores
        scores += self._calculate_kang_scores()
        
        return scores

    def _get_intermediate_reward(self, player) -> float:
        """Calculates engineered intermediate rewards to deal with the sparse reward problem.
        This function is called at every non-terminal step and makes it easier for an agent to learn the environment.
        """
        # The READY_DIVIDER value divides the score for a winning combination a player is ready for.
        # Example: being ready for a basic win worth 5 points gives an intermediate reward of 5/READY_DIVIDER
        READY_DIVIDER = 10
        # The multipliers represent how much different things like a pong or pair are worth.
        # Example: having two families and one pair is worth 2*FAMILY_MULTIPLIER + 1*PAIR_MULTIPLIER
        KANG_MULTIPLIER = 0.3
        PONG_MULTIPLIER = 0.08 # A pong is not as much worth as a three-of-a-kind, as other players can see the pong cards
        THREE_OF_A_KIND_MULTIPLIER = 0.1 # Three-of-a-kinds are a bit better than straights because they are used in special combinations and they can lead to kangs
        STRAIGHT_MULTIPLIER = 0.08
        PAIR_MULTIPLIER = 0.05

        # If the player is ready return an intermediate reward based on being ready and kangs
        if ready_combination_score := self._is_ready(player):
            # Calculate points for kangs
            number_of_kangs = len(np.nonzero(self._board[player]["kangs"])[0])
            kang_score = number_of_kangs * KANG_MULTIPLIER
            return ready_combination_score / READY_DIVIDER + kang_score
        else: # If a player is not ready, return an intermediate reward based on pongs, pairs.. etc.

            # The player's hand
            hand = self._board[player]["hand"].copy()

            # Count the number of three-of-a-kinds and pairs in the player's hand 
            number_of_three_of_a_kinds = len(np.where(hand==3)[0])
            number_of_pairs = len(np.where(hand==2)[0])

            # Count the number of straights in the player's hand
            number_of_straights = 0
            while True:
                for i in range(len(hand)-2):
                    if hand[i] and hand[i+1] and hand[i+2]:
                        number_of_straights += 1
                        hand[[i, i+1, i+2]] -= 1
                        break
                break

            # Count the number of pongs
            number_of_pongs = len(np.where(self._board[player]["pongs"]==3)[0])

            return (number_of_pongs*PONG_MULTIPLIER +
                    number_of_three_of_a_kinds*THREE_OF_A_KIND_MULTIPLIER +
                    number_of_straights*STRAIGHT_MULTIPLIER +
                    number_of_pairs*PAIR_MULTIPLIER)

    def get_valid_actions(self):
        return np.nonzero(self._board[self._player_turn]["hand"])[0]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random | <- this comment and line is from the gymnasium documentation
        super().reset(seed=seed)

        # Reset the deck
        self._deck = [n for n in range(self._UNIQUE_CARDS)]*self._DUPLICATES
        # Shuffle the deck
        np.random.shuffle(self._deck)

        # Deal cards to players
        for player in range(self._NUMBER_OF_PLAYERS):
            cards = self._deck[:self._HAND_SIZE-1] # deal 13 cards
            self._deck = self._deck[self._HAND_SIZE-1:] # remove the cards from the deck
            hand = np.eye(self._UNIQUE_CARDS)[cards].sum(axis=0, dtype=int) # one-hot-encode cards and sum the axis. 

            assert len(hand) == self._UNIQUE_CARDS
            assert np.max(hand) <= self._DUPLICATES

            self._board[player] = {
                "hand": hand,
                "pongs": np.zeros(self._UNIQUE_CARDS, dtype=int),
                "kangs": np.zeros(self._UNIQUE_CARDS, dtype=int),
                "discarded": np.zeros(self._UNIQUE_CARDS, dtype=int),
            }

        # Sets the turn
        self._player_turn = self._AI
        # Give one extra card to player who starts
        self._board[self._player_turn]["hand"][self._deck.pop()] += 1

        # Set flag
        self._game_running = True
        # Reset the game scores
        self._game_result = np.zeros(self._NUMBER_OF_PLAYERS)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        assert self._game_running
        assert self._board[self._player_turn]["hand"][action] != 0 # check that the action is valid

        # The player discards a card
        discarded_card = action
        self._board[self._player_turn]["hand"][discarded_card] -= 1

        # Check if any player can take the discarded card to win
        for player in range(self._NUMBER_OF_PLAYERS):
            if player == self._player_turn:
                continue
            combination_score = self._is_winning_combination(player, discarded_card) # does the player need the discarded card to win
            # A player can only take the discarded card if they need it for a special winning combination or if they have a kang.
            if (combination_score > self._BASIC_WIN_SCORE) or (combination_score and self._board[player]["kang"].sum() > 0):
                # A player takes the discarded card and gets a winning combination
                self._game_running = False
                self._player_turn = player # Set turn to the player who won
                # Return
                terminated = True
                truncated = False
                self._game_result = self._calculate_win_scores(self._player_turn, combination_score, self._player_turn)
                reward = self._game_result[self._AI]
                observation = self._get_obs()
                info = self._get_info()
                return observation, reward, terminated, truncated, info

        # Check if any player can take the discarded card for pong or kang
        for player in range(self._NUMBER_OF_PLAYERS):
            if player == self._player_turn:
                continue

            # If a player has two or more of the discarded card he can take it
            if self._board[player]["hand"][discarded_card] >= 2:
                self._player_turn = player # Set turn to the player who takes the discarded card
                if self._board[player]["hand"][discarded_card] == 3 and len(self._deck) > 0: # kang (but is only allowed if there is at least one card left in the deck)
                    # Remove the cards from the hand and set the kang. (the 3 cards from the hand and the discarded card form the kang)
                    self._board[player]["hand"][discarded_card] -= 3
                    self._board[player]["kangs"][discarded_card] = 1 # set the kang

                    # Give the player who kanged an extra card (since 4 cards is used for the "kang family")
                    dealt_card = self._deck.pop()
                    self._board[self._player_turn]["hand"][dealt_card] += 1

                    # Check if the player has won after being dealt the card
                    combination_score = self._is_winning_combination(self._player_turn, dealt_card)
                    self._game_running = not combination_score
                else: # pong
                    # Remove the cards from the hand and set the pong. (the 2 cards from the hand and the discarded card form the pong)
                    self._board[player]["hand"][discarded_card] -= 2
                    self._board[player]["pongs"][discarded_card] = 1 # set the pong

                # Return
                terminated = not self._game_running
                truncated = False
                if terminated:
                    self._game_result = self._calculate_win_scores(self._player_turn, combination_score, self._player_turn)
                    reward = self._game_result[self._AI]
                else:
                    reward = self._get_intermediate_reward(self._AI)
                observation = self._get_obs()
                info = self._get_info()
                if terminated or self._player_turn == self._AI:
                    return observation, reward, terminated, truncated, info
                else:
                    return self.step(np.random.choice(self.get_valid_actions()))

        # Add to the player's discarded cards
        self._board[self._player_turn]["discarded"][discarded_card] += 1

        # Change turn to next player
        self._player_turn = (self._player_turn + 1) % self._NUMBER_OF_PLAYERS

        # Game over if there are no cards left in the deck
        if len(self._deck) == 0:
            self._game_running = False
            combination_score = 0
        else:
            # The next player gets dealt one card
            dealt_card = self._deck.pop()
            self._board[self._player_turn]["hand"][dealt_card] += 1

            # Check if the next player has won
            combination_score = self._is_winning_combination(self._player_turn, dealt_card)
            self._game_running = not combination_score

        terminated = not self._game_running
        truncated = False
        if terminated:
            self._game_result = self._calculate_win_scores(self._player_turn, combination_score, self._player_turn) if combination_score else self._calculate_draw_scores()
            reward = self._game_result[self._AI]
        else:
            reward = self._get_intermediate_reward(self._AI)
        observation = self._get_obs()
        info = self._get_info()

        if terminated or self._player_turn == self._AI:
            return observation, reward, terminated, truncated, info
        else:
            return self.step(np.random.choice(self.get_valid_actions()))
    
gym.register(
    id="Mahjong-v0",
    entry_point=MahjongEnv,
)