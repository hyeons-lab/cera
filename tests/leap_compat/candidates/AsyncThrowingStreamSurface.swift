// Deliberately incompatible compile-only candidate, never a production module.
public protocol MessageResponse {}
public class GenerationOptions {}
public typealias SkieSwiftFlow<T> = AsyncThrowingStream<T, Error>
public protocol Conversation {
  func generateResponse(
    userTextMessage: String, generationOptions: GenerationOptions?
  ) -> SkieSwiftFlow<any MessageResponse>
}
