from search import answer_question

EXIT_COMMANDS = ("sair", "exit", "quit")


def main() -> None:
    print("Chat iniciado. Digite sua pergunta ou 'sair' para encerrar.")
    print()

    while True:
        question = input("PERGUNTA: ").strip()
        if question.lower() in EXIT_COMMANDS:
            print("Encerrando o chat. Até mais!")
            break
        if not question:
            continue

        try:
            answer = answer_question(question)
        except Exception as error:
            print(
                "Não foi possível responder. "
                f"Verifique se o banco está ativo e a API key configurada: {error}"
            )
            continue

        print(f"RESPOSTA: {answer}")
        print()


if __name__ == "__main__":
    main()
