// Deliberately partial compile-only candidate, never a production module.
public protocol MessageResponse {}
public class GenerationOptions {}
public typealias SkieSwiftFlow<T> = AsyncStream<T>
public protocol Conversation {
  func generateResponse(
    userTextMessage: String, generationOptions: GenerationOptions?
  ) -> SkieSwiftFlow<any MessageResponse>
}
